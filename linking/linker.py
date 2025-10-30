from pathlib import Path
import logging, json, numpy as np, csv, time
from rdflib import Graph, Namespace, URIRef, RDFS
from torch import tensor
from sentence_transformers import SentenceTransformer, util
import os
from nltk.corpus import wordnet as wn
import Levenshtein
from rapidfuzz import process, fuzz
from functools import lru_cache
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
kg_path = Path('linking/wn_full.owl')
definition_iri = "http://example.org/wordnet.owl#definition"
identifier_iri = "http://example.org/wordnet.owl#identifier"
IMGNET_SOURCE_PATH = 'experiments/perceived-entity-linking/imagenet.txt'
VGENOME_SOURCE_PATH = 'experiments/perceived-entity-linking/visual_genome.json'
EX = Namespace("http://example.org/wordnet.owl#")

def load_groundtruth(source_path):
    logging.info(f"Loading ground truth from {source_path}...")
    ext = os.path.splitext(source_path)[1].lower()
    results = {}

    if ext == ".json":
        with open(source_path, "r") as f:
            data = json.load(f)

        for phrase, synset_name in data.items():
            try:
                syn = wn.synset(synset_name)
                id_corrected = f"{syn.offset():08d}-{syn.pos()}"
                results.setdefault(id_corrected, []).append(phrase)
            except Exception as e:
                logging.warning(f"Skipping '{phrase}': invalid synset '{synset_name}' ({e})")

    elif ext == ".txt":
        with open(source_path, "r") as f:
            for concept in f:
                concept = concept.strip()
                if not concept:
                    continue
                id_ = concept.split(' ')[0]
                id_corrected = id_[1:] + '-n'
                names = concept.split(id_)[1].strip().split(', ')
                results[id_corrected] = names
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    logging.debug(f"Loaded {len(results)} concepts.")
    return results

# =============================
#  Perceived-Entity Linker
# -----------------------------

class PerceivedEntityLinker:
    def __init__(self, kg_path='linking/wn_full.owl', model="all-MiniLM-L6-v2"):
        self.kg_path = Path(kg_path)
        self.definition_iri = "http://example.org/wordnet.owl#definition"
        self.altlabel_iri = "http://example.org/wordnet.owl#altLabel"
        self.embedder = SentenceTransformer(model)
        self.concepts, self.kg, self.id_to_label, self.label_to_iri = self.load_knowledgegraph()
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.physical_descendants = self.get_descendants(EX["C_00001930-n"])
        self.target_emb = None
        logging
        if Path("embedded_kg.npz").exists():
            self.embedded_kg = self.load_embedded_kg()
        else:
            self.embedded_kg = self.embed_knowledgegraph(self.concepts)
            self.save_embedded_kg()
            
        logging.info("PerceivedEntityLinker initialized and ready.")

    def load_knowledgegraph(self):
        logging.info("Loading knowledge graph via SPARQL...")
        g = Graph()
        g.parse(str(self.kg_path))

        q = f"""
        SELECT ?label ?id ?def ?c WHERE {{
            ?c rdfs:label ?label .
            OPTIONAL {{ ?c <{identifier_iri}> ?id . }}
            OPTIONAL {{ ?c <{definition_iri}> ?def . }}
        }}
        """
        results = list(g.query(q))  # materialize once
        logging.debug("Loaded graph...")
        concept_id_lookup = {
            str(r['id']): f"{r['label']}. Definition: {r['def'] if r['def'] else ''}"
            for r in results
            if r['id'] is not None
        }
        concept_iri_lookup = {
            f"{r['label']}. Definition: {r['def'] if r['def'] else ''}": str(r['c'])
            for r in results
        }
        concepts = list(concept_iri_lookup.keys())
        logging.debug(f"Loaded {len(concepts)} concepts. Example: {concepts[:3]}")
        return concepts, g, concept_id_lookup, concept_iri_lookup

    def get_altlabels_by_label(self, label):
        q = f"""
        SELECT DISTINCT ?l ?altlabel WHERE {{
            ?entity rdfs:label ?l .
            FILTER (lcase(str(?l)) = lcase("{label}"))
            OPTIONAL {{ ?entity <{self.altlabel_iri}> ?altlabel . }}
        }}
        """
        results = list(self.kg.query(q))

        if not results:
            logging.debug(f"No entity found for label: {label}")
            return [label]
        altlabels = {str(r["altlabel"]) for r in results if r["altlabel"]}

        return list(altlabels).append(label)

    def get_descendants(self, root):
        q = f"""
        SELECT ?x WHERE {{
            ?x rdfs:subClassOf* <{root}> .
        }}
        """
        return {str(row[0]) for row in self.kg.query(q)}

    def get_hierarchy(self, iri, depth=2):
        """
        Returns a concatenated string containing the labels and optional definitions
        of the given entity and its ancestors up to the specified depth.
        """
        context = []
        current_level = {URIRef(iri)}
        visited = set()

        for _ in range(depth):
            next_level = set()
            for e in current_level:
                if e in visited:
                    continue
                visited.add(e)

                # --- Fetch label and optional definition in one query ---
                q = f"""
                SELECT ?l ?def WHERE {{
                    OPTIONAL {{ <{e}> rdfs:label ?l . }}
                    OPTIONAL {{ <{e}> <{self.definition_iri}> ?def . }}
                }}
                """
                for row in self.kg.query(q):
                    label = str(row["l"]) if row["l"] else ""
                    definition = str(row["def"]) if row["def"] else ""
                    # Add both if present
                    if label or definition:
                        context.append(f"{label}. {definition}")

                # --- Get direct parents ---
                for parent_row in self.kg.query(
                    f"SELECT ?p WHERE {{ <{e}> rdfs:subClassOf ?p }}"
                ):
                    next_level.add(parent_row["p"])
            current_level = next_level
        return " ".join(context)
    
    def embed_knowledgegraph(self, concepts):
        logging.info("Embedding knowledge graph...")
        embeddings = self.embedder.encode(concepts, convert_to_tensor=True)
        self.embedded_kg = dict(zip(concepts, embeddings))
        logging.debug(f"Embedding shape: {embeddings.shape}")
        return self.embedded_kg
    
    def save_embedded_kg(self, path="embedded_kg.npz", meta_path="embedded_kg_keys.json"):
        keys, vecs = zip(*[(k, v.cpu().numpy()) for k, v in self.embedded_kg.items()])
        np.savez_compressed(path, *vecs)
        json.dump(keys, open(meta_path, "w"))
        logging.info(f"Saved {len(keys)} embeddings → {path}")

    def load_embedded_kg(self, path="embedded_kg.npz", meta_path="embedded_kg_keys.json"):
        keys = json.load(open(meta_path))
        arrs = np.load(path)
        self.embedded_kg = {k: tensor(arrs[f"arr_{i}"]) for i, k in enumerate(keys)}
        logging.info(f"Loaded {len(self.embedded_kg)} embeddings from disk.")
        return self.embedded_kg
    
    def embed_single_entity(self, string):
        logging.info(f"Embedding entity: {string}")
        emb = self.embedder.encode(string, convert_to_tensor=True)
        return emb

    def knn_retrieval(self, embedded_entity, k=5):
        scores = {key: util.cos_sim(embedded_entity, v).item() for key, v in self.embedded_kg.items()}
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return top_k

    def candidate_selection(self, concept, k=5):
        start = time.time()
        self.target_emb = self.embed_single_entity(concept)
        closests = self.knn_retrieval(self.target_emb, k)
        elapsed = round(time.time() - start, 2)
        logging.debug(f"Closest match for '{concept}' → '{closests[0][0]}' (time: {elapsed}s)")
        return closests, elapsed

    def check_physical_entity_sparql(self, entity_iri):
        return str(entity_iri) in self.physical_descendants


    def disambiguate(self, candidates, depth=2):
        scores = []
        start = time.time()
        for desc, base_score in candidates:
            # --- Type filters ---
            delt_noun = 1 if self.label_to_id.get(desc)[-1] == 'n' else 0
            iri = self.label_to_iri[desc]
            delt_phys = 1 if str(iri) in self.physical_descendants else 0
            mask = delt_noun * delt_phys

            # --- Hierarchical context ---
            descendants = self.get_hierarchy(iri, depth=depth)
            combined_text = desc + " " + descendants
            vec = self.embedder.encode(combined_text)

            # --- Similarity computation ---
            sim = util.cos_sim(self.target_emb, vec).item()  # scalar

            # --- Masked final score ---
            final_score = mask * sim
            scores.append((desc, final_score))

            logging.debug(
                f"Disambiguation: {desc}, mask={mask}, sim={sim:.3f}, "
                f"descendants='{descendants[:80]}...'"
            )
        elapsed = round(time.time() - start, 2)
        return sorted(scores, key=lambda x: x[1], reverse=True), elapsed
    
# =============================
#  Ablated Perceived-Entity Linker
# -----------------------------
class AblatedPerceivedEntityLinker:
    def __init__(self, condition='', kg_path='linking/wn_full.owl', model="all-MiniLM-L6-v2"):
        self.kg_path = Path(kg_path)
        self.condition = condition #no-context, no-physical, no-noun
        self.definition_iri = "http://example.org/wordnet.owl#definition"
        self.altlabel_iri = "http://example.org/wordnet.owl#altLabel"
        self.embedder = SentenceTransformer(model)
        self.concepts, self.kg, self.id_to_label, self.label_to_iri = self.load_knowledgegraph()
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.physical_descendants = self.get_descendants(EX["C_00001930-n"])
        self.target_emb = None
        self.ALT   = URIRef(self.altlabel_iri)
        self.altlabels    : dict[str,set] = {}
        if Path("embedded_kg.npz").exists():
            self.embedded_kg = self.load_embedded_kg()
        else:
            self.embedded_kg = self.embed_knowledgegraph(self.concepts)
            self.save_embedded_kg()
        for s, p, o in self.kg.triples((None, None, None)):
            s_str = str(s)
        if p == self.ALT:
                self.altlabels.setdefault(s_str, set()).add(str(o))            
        logging.info("PerceivedEntityLinker initialized and ready.")

    def get_altlabels_by_label(self, label):
        # Resolve rdfs:label -> IRI -> altLabels; include the input label itself; dedup via set.
        iri = self.label_to_iri.get(label.lower())
        if not iri:
            return [label]
        return list(self.altlabels.get(iri, set()) | {label})
    
    def load_knowledgegraph(self):
        logging.info("Loading knowledge graph via SPARQL...")
        g = Graph()
        g.parse(str(self.kg_path))

        q = f"""
        SELECT ?label ?id ?def ?c WHERE {{
            ?c rdfs:label ?label .
            OPTIONAL {{ ?c <{identifier_iri}> ?id . }}
            OPTIONAL {{ ?c <{definition_iri}> ?def . }}
        }}
        """
        results = list(g.query(q))  # materialize once
        logging.debug("Loaded graph...")

        concept_id_lookup = {
            str(r['id']): f"{r['label']}. Definition: {r['def'] if r['def'] else ''}"
            for r in results
            if r['id'] is not None
        }
        concept_iri_lookup = {
            f"{r['label']}. Definition: {r['def'] if r['def'] else ''}": str(r['c'])
            for r in results
        }
        concepts = list(concept_iri_lookup.keys())
        logging.debug(f"Loaded {len(concepts)} concepts. Example: {concepts[:3]}")
        return concepts, g, concept_id_lookup, concept_iri_lookup

    def get_descendants(self, root):
        q = f"""
        SELECT ?x WHERE {{
            ?x rdfs:subClassOf* <{root}> .
        }}
        """
        return {str(row[0]) for row in self.kg.query(q)}

    def get_hierarchy(self, iri, depth=2):
        """
        Returns a concatenated string containing the labels and optional definitions
        of the given entity and its ancestors up to the specified depth.
        """
        context = []
        current_level = {URIRef(iri)}
        visited = set()

        for _ in range(depth):
            next_level = set()
            for e in current_level:
                if e in visited:
                    continue
                visited.add(e)

                # --- Fetch label and optional definition in one query ---
                q = f"""
                SELECT ?l ?def WHERE {{
                    OPTIONAL {{ <{e}> rdfs:label ?l . }}
                    OPTIONAL {{ <{e}> <{self.definition_iri}> ?def . }}
                }}
                """
                for row in self.kg.query(q):
                    label = str(row["l"]) if row["l"] else ""
                    definition = str(row["def"]) if row["def"] else ""
                    # Add both if present
                    if label or definition:
                        context.append(f"{label}. {definition}")

                # --- Get direct parents ---
                for parent_row in self.kg.query(
                    f"SELECT ?p WHERE {{ <{e}> rdfs:subClassOf ?p }}"
                ):
                    next_level.add(parent_row["p"])
            current_level = next_level
        return " ".join(context)
    
    def embed_knowledgegraph(self, concepts):
        logging.info("Embedding knowledge graph...")
        embeddings = self.embedder.encode(concepts, convert_to_tensor=True)
        self.embedded_kg = dict(zip(concepts, embeddings))
        logging.debug(f"Embedding shape: {embeddings.shape}")
        return self.embedded_kg
    
    def save_embedded_kg(self, path="embedded_kg.npz", meta_path="embedded_kg_keys.json"):
        keys, vecs = zip(*[(k, v.cpu().numpy()) for k, v in self.embedded_kg.items()])
        np.savez_compressed(path, *vecs)
        json.dump(keys, open(meta_path, "w"))
        logging.info(f"Saved {len(keys)} embeddings → {path}")

    def load_embedded_kg(self, path="embedded_kg.npz", meta_path="embedded_kg_keys.json"):
        keys = json.load(open(meta_path))
        arrs = np.load(path)
        self.embedded_kg = {k: tensor(arrs[f"arr_{i}"]) for i, k in enumerate(keys)}
        logging.info(f"Loaded {len(self.embedded_kg)} embeddings from disk.")
        return self.embedded_kg
    
    def embed_single_entity(self, string):
        logging.info(f"Embedding entity: {string}")
        emb = self.embedder.encode(string, convert_to_tensor=True)
        return emb

    def knn_retrieval(self, embedded_entity, k=5):
        scores = {key: util.cos_sim(embedded_entity, v).item() for key, v in self.embedded_kg.items()}
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return top_k

    def candidate_selection(self, concept, k=5):
        start = time.time()
        self.target_emb = self.embed_single_entity(concept)
        closests = self.knn_retrieval(self.target_emb, k)
        elapsed = round(time.time() - start, 2)
        logging.debug(f"Closest match for '{concept}' → '{closests[0][0]}' (time: {elapsed}s)")
        return closests, elapsed

    def check_physical_entity_sparql(self, entity_iri):
        return str(entity_iri) in self.physical_descendants


    def disambiguate(self, candidates, depth=2):
        scores = []
        start = time.time()
        for desc, base_score in candidates:
            iri = self.label_to_iri[desc]
            # --- Type filters ---
            if self.condition == 'no-physical':
                mask = 1 if self.label_to_id.get(desc)[-1] == 'n' else 0        
            if self.condition == 'no-noun':
                mask = 1 if str(iri) in self.physical_descendants else 0  
            else:
                delt_noun = 1 if self.label_to_id.get(desc)[-1] == 'n' else 0
                delt_phys = 1 if str(iri) in self.physical_descendants else 0
                mask = delt_noun * delt_phys

            if self.condition != 'no-context':
                # --- Hierarchical context ---
                if self.condition == 'altlabel-context':
                    altlabels = self.get_altlabels_by_label(desc)
                    print(f"Concept {desc} has alternative labels: {altlabels}\n{'*'*100}")
                    
                    # --- collect embeddings ---
                    all_labels = [desc] + altlabels
                    all_embeddings = [self.embed_single_entity(lbl) for lbl in set(all_labels)]

                    # --- average embeddings along axis 0 ---
                    vec = sum(all_embeddings) / len(all_embeddings)
                else:
                    descendants = self.get_hierarchy(iri, depth=depth)
                    combined_text = desc + " " + descendants
                    vec = self.embedder.encode(combined_text)

                # --- Similarity computation ---
                sim = util.cos_sim(self.target_emb, vec).item()  # scalar
            else:
                sim = base_score
            # --- Masked final score ---
            final_score = mask * sim
            scores.append((desc, final_score))

            logging.debug(
                f"Disambiguation: {desc}, mask={mask}, sim={sim:.3f}, "
                # f"descendants='{descendants[:80]}...'"
            )
        elapsed = round(time.time() - start, 2)
        return sorted(scores, key=lambda x: x[1], reverse=True), elapsed

# =================================
#  BaseLine Perceived-Entity Linker
# ---------------------------------


class BaseLinePerceivedEntityLinker:
    def __init__(self, kg_path="linking/wn_full.owl"):
        # --- config + well-typed URIs (use URIRef; string equality fails for rdflib terms) ---
        self.kg_path = Path(kg_path)
        self.definition_iri  = "http://example.org/wordnet.owl#definition"
        self.identifier_iri  = "http://example.org/wordnet.owl#identifier"
        self.altlabel_iri    = "http://example.org/wordnet.owl#altLabel"
        self.target_concept  = None

        self.ALT   = URIRef(self.altlabel_iri)          # altLabel predicate
        self.IDENT = URIRef(self.identifier_iri)        # identifier predicate

        # --- load KG once ---
        self.kg = Graph()
        self.kg.parse(str(self.kg_path))

        # --- precompute in-memory indices for O(1) lookups ---
        # Note: use IRIs (stringified) as canonical keys to avoid term-type surprises.
        self.label_to_iri : dict[str,str] = {}          # "apple" -> "http://.../Apple_iri"
        self.iri_to_label : dict[str,str] = {}          # "http://.../Apple_iri" -> "Apple"
        self.altlabels    : dict[str,set] = {}          # "http://.../Apple_iri" -> {"pome", "malus domestica", ...}
        self.subclasses   : dict[str,set] = {}          # "child_iri" -> {"parent_iri1", "parent_iri2", ...}
        self.iri_to_id    : dict[str,str] = {}          # "http://.../Apple_iri" -> "wn:01234567-n"
        self.id_to_label  : dict[str,str] = {}          # "wn:01234567-n" -> "Apple"

        # --- single pass over triples; compare predicates to RDFS/URIRef (not str) ---
        for s, p, o in self.kg.triples((None, None, None)):
            s_str = str(s)

            # rdfs:label
            if p == RDFS.label:
                lbl = str(o)
                self.iri_to_label[s_str] = lbl
                self.label_to_iri[lbl.lower()] = s_str

            # altLabel
            elif p == self.ALT:
                self.altlabels.setdefault(s_str, set()).add(str(o))

            # rdfs:subClassOf
            elif p == RDFS.subClassOf:
                self.subclasses.setdefault(s_str, set()).add(str(o))

            # identifier
            elif p == self.IDENT:
                self.iri_to_id[s_str] = str(o)

        # --- backfill id -> label once labels are known (2nd pass to avoid order dependency) ---
        for iri, _id in self.iri_to_id.items():
            lbl = self.iri_to_label.get(iri)
            if lbl is not None:
                self.id_to_label[_id] = lbl

        # --- optional convenience map (label -> id); safe only when both exist ---
        self.label_to_id = {}
        for _id, lbl in self.id_to_label.items():
            self.label_to_id[lbl] = _id

        logging.info(f"Preloaded {len(self.label_to_iri)} labels, "
                     f"{sum(len(v) for v in self.altlabels.values())} altlabels, "
                     f"{sum(len(v) for v in self.subclasses.values())} subclass links, "
                     f"{len(self.iri_to_id)} identifiers")

    # --- caching + fast helpers (unchanged logic; comments explain intent) ---
    @lru_cache(maxsize=None)
    def get_synset(self, name):
        # Normalize for WN lookup; cache to avoid repeated I/O-bound calls.
        query = name.strip().replace(" ", "_").lower()
        return wn.synsets(query)

    @lru_cache(maxsize=None)
    def get_names(self, synset):
        # Lemma names of a synset; cached because we may revisit same synsets.
        return synset.lemma_names() if synset else []

    def get_altlabels_by_label(self, label):
        # Resolve rdfs:label -> IRI -> altLabels; include the input label itself; dedup via set.
        iri = self.label_to_iri.get(label.lower())
        if not iri:
            return [label]
        return list(self.altlabels.get(iri, set()) | {label})

    def get_entities_by_label(self, label):
        # Resolve label directly to entity IRI if present; keep API compatibility (list).
        iri = self.label_to_iri.get(label.lower())
        return [iri] if iri else []

    def get_class(self, entity_iri):
        # Return direct rdfs:subClassOf parents (as IRIs); empty list if none.
        return list(self.subclasses.get(entity_iri, []))

    def candidate_selection(self, concept, k=5):
        # Build candidate entity IRIs via altLabels+lemmas; score by RapidFuzz and pick top-k.
        start = time.time()
        self.target_concept = concept
        synsets = self.get_synset(concept)
        if not synsets:
            return [(concept, 1.0)], round(time.time() - start, 2)

        candidates = []
        altlabels = self.get_altlabels_by_label(concept)
        for sn in synsets:
            names = self.get_names(sn)
            for name in set(names + altlabels):
                sim = fuzz.ratio(concept, name) / 100.0  # normalize 0..1
                candidates.append((name, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_entities = []
        for c, _ in candidates[:k]:
            top_entities.extend(self.get_entities_by_label(c))
        return top_entities[:k], round(time.time() - start, 2)

    def get_synset_similarity(self, synsets_A, synset_B):
        # Max string similarity across lemma-names of synsets; cheap and robust baseline.
        if not synsets_A or not synset_B:
            return 0.0
        target = synset_B.name().split(".")[0].replace("_", " ").lower()
        return max(
            fuzz.ratio(s.name().split(".")[0].replace("_", " ").lower(), target) / 100.0
            for s in synsets_A
        )

    def disambiguate(self, candidates, target_concept=None):
        # Make disambiguation independent of prior call order; allow explicit target.
        if target_concept is not None:
            self.target_concept = target_concept
        if not self.target_concept:
            raise ValueError("target_concept must be provided or set earlier via candidate_selection().")

        start = time.time()
        target_ss = self.get_synset(self.target_concept)
        if not target_ss:
            return candidates, 0.0

        scores = []
        for c in candidates:
            high_score = 0.0
            for cls in self.get_class(c):
                for ss in self.get_synset(cls):
                    sim = self.get_synset_similarity(target_ss, ss)
                    high_score = max(high_score, sim)
            scores.append((c, high_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores, round(time.time() - start, 2)


# ========================================================================================
# EXPERIMENTS
# ----------------------------------------------------------------------------------------

def experiment_baseline(groundtruth, source_name):
    
    linker = BaseLinePerceivedEntityLinker()

    rows = []

    for i, target_id in enumerate(list(groundtruth.keys())):
        names = groundtruth[target_id]
        for name in names:
            closests, elapsed_cand = linker.candidate_selection(name)
            closests_ordered, elapsed_disamb = linker.disambiguate(closests)
            # print(closests_ordered)
            if closests_ordered:
                closest = closests_ordered[0][0]
                clean_result = closest.split('. Definition')[0].strip()
                result_id = linker.iri_to_id.get(closest, 'N/A')
            else:
                clean_result = 'N/A'
                result_id = 'N/A'
            correct = (target_id == result_id)
            logging.info(
                f"[{i+1}] {name}: Target={target_id}, Result={result_id}, "
                f"LabelMatch={clean_result}, Match={correct}"
            )
            rows.append({
                "query": name,
                "closest_label": clean_result,
                "target_id": target_id,
                "result_id": result_id,
                "correct": correct,
                "time_sec_candidate_selection": elapsed_cand,
                "time_sec_disambiguation": elapsed_disamb
            })
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-baseline-{source_name}.csv')
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")

def experiment_orvis_linker(groundtruth, source_name, depth=1):
    linker = PerceivedEntityLinker()
    rows = []
    for i, target_id in enumerate(list(groundtruth.keys())):
        names = groundtruth[target_id]
        for name in names:
            closests, elapsed_cand = linker.candidate_selection(name)
            closests_ordered, elapsed_disamb = linker.disambiguate(closests, depth)
            
            if closests_ordered:
                closest = closests_ordered[0][0]
                clean_result = closest.split('. Definition')[0].strip()
                print(closest[0])
                result_id = linker.label_to_id.get(closest, 'N/A')
            else:
                clean_result = 'N/A'
                result_id = 'N/A'
            correct = (target_id == result_id)
            logging.info(
                f"[{i+1}] {name}: Target={target_id}, Result={result_id}, "
                f"LabelMatch={clean_result}, Match={correct}"
            )
            rows.append({
                "query": name,
                "closest_label": clean_result,
                "target_id": target_id,
                "result_id": result_id,
                "correct": correct,
                "time_sec_candidate_selection": elapsed_cand,
                "time_sec_disambiguation": elapsed_disamb
            })
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-complete-depth{depth}-{source_name}.csv')
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")

def experiment_orvis_linker_no_disambiguation(groundtruth, source_name):
    linker = PerceivedEntityLinker()
    rows = []
    for i, target_id in enumerate(list(groundtruth.keys())):
        names = groundtruth[target_id]
        for name in names:
            closests, elapsed_cand = linker.candidate_selection(name)
            closests_ordered, elapsed_disamb = closests, 0
            
            if closests_ordered:
                closest = closests_ordered[0][0]
                clean_result = closest.split('. Definition')[0].strip()
                print(closest[0])
                result_id = linker.label_to_id.get(closest, 'N/A')
            else:
                clean_result = 'N/A'
                result_id = 'N/A'
            correct = (target_id == result_id)
            logging.info(
                f"[{i+1}] {name}: Target={target_id}, Result={result_id}, "
                f"LabelMatch={clean_result}, Match={correct}"
            )
            rows.append({
                "query": name,
                "closest_label": clean_result,
                "target_id": target_id,
                "result_id": result_id,
                "correct": correct,
                "time_sec_candidate_selection": elapsed_cand,
                "time_sec_disambiguation": elapsed_disamb
            })
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-no_disamb-{source_name}.csv')
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")

def experiment_orvis_linker_no_context(groundtruth, source_name):
    condition = 'no-context'
    linker = AblatedPerceivedEntityLinker(condition=condition)
    rows = []
    for i, target_id in enumerate(list(groundtruth.keys())):
        names = groundtruth[target_id]
        for name in names:
            closests, elapsed_cand = linker.candidate_selection(name)
            closests_ordered, elapsed_disamb = linker.disambiguate(closests)
            
            if closests_ordered:
                closest = closests_ordered[0][0]
                clean_result = closest.split('. Definition')[0].strip()
                print(closest[0])
                result_id = linker.label_to_id.get(closest, 'N/A')
            else:
                clean_result = 'N/A'
                result_id = 'N/A'
            correct = (target_id == result_id)
            logging.info(
                f"[{i+1}] {name}: Target={target_id}, Result={result_id}, "
                f"LabelMatch={clean_result}, Match={correct}"
            )
            rows.append({
                "query": name,
                "closest_label": clean_result,
                "target_id": target_id,
                "result_id": result_id,
                "correct": correct,
                "time_sec_candidate_selection": elapsed_cand,
                "time_sec_disambiguation": elapsed_disamb
            })
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-{condition}-{source_name}.csv')
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")

def experiment_orvis_linker_no_noun_filter(groundtruth, source_name):
    condition = 'no-noun'
    linker = AblatedPerceivedEntityLinker(condition=condition)
    rows = []
    for i, target_id in enumerate(list(groundtruth.keys())):
        names = groundtruth[target_id]
        for name in names:
            closests, elapsed_cand = linker.candidate_selection(name)
            closests_ordered, elapsed_disamb = linker.disambiguate(closests)
            
            if closests_ordered:
                closest = closests_ordered[0][0]
                clean_result = closest.split('. Definition')[0].strip()
                print(closest[0])
                result_id = linker.label_to_id.get(closest, 'N/A')
            else:
                clean_result = 'N/A'
                result_id = 'N/A'
            correct = (target_id == result_id)
            logging.info(
                f"[{i+1}] {name}: Target={target_id}, Result={result_id}, "
                f"LabelMatch={clean_result}, Match={correct}"
            )
            rows.append({
                "query": name,
                "closest_label": clean_result,
                "target_id": target_id,
                "result_id": result_id,
                "correct": correct,
                "time_sec_candidate_selection": elapsed_cand,
                "time_sec_disambiguation": elapsed_disamb
            })
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-{condition}-{source_name}.csv')
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")

def experiment_orvis_linker_no_physical_filter(groundtruth, source_name):
    condition = 'no-physical'
    linker = AblatedPerceivedEntityLinker(condition=condition)
    rows = []
    for i, target_id in enumerate(list(groundtruth.keys())):
        names = groundtruth[target_id]
        for name in names:
            closests, elapsed_cand = linker.candidate_selection(name)
            closests_ordered, elapsed_disamb = linker.disambiguate(closests)
            
            if closests_ordered:
                closest = closests_ordered[0][0]
                clean_result = closest.split('. Definition')[0].strip()
                print(closest[0])
                result_id = linker.label_to_id.get(closest, 'N/A')
            else:
                clean_result = 'N/A'
                result_id = 'N/A'
            correct = (target_id == result_id)
            logging.info(
                f"[{i+1}] {name}: Target={target_id}, Result={result_id}, "
                f"LabelMatch={clean_result}, Match={correct}"
            )
            rows.append({
                "query": name,
                "closest_label": clean_result,
                "target_id": target_id,
                "result_id": result_id,
                "correct": correct,
                "time_sec_candidate_selection": elapsed_cand,
                "time_sec_disambiguation": elapsed_disamb
            })
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-{condition}-{source_name}.csv')
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")


def experiment_orvis_linker_altlabel_context(groundtruth, source_name):
    condition = 'altlabel-context'
    linker = AblatedPerceivedEntityLinker(condition=condition)
    rows = []
    for i, target_id in enumerate(list(groundtruth.keys())):
        names = groundtruth[target_id]
        for name in names:
            closests, elapsed_cand = linker.candidate_selection(name)
            closests_ordered, elapsed_disamb = linker.disambiguate(closests)
            
            if closests_ordered:
                closest = closests_ordered[0][0]
                clean_result = closest.split('. Definition')[0].strip()
                print(closest[0])
                result_id = linker.label_to_id.get(closest, 'N/A')
            else:
                clean_result = 'N/A'
                result_id = 'N/A'
            correct = (target_id == result_id)
            logging.info(
                f"[{i+1}] {name}: Target={target_id}, Result={result_id}, "
                f"LabelMatch={clean_result}, Match={correct}"
            )
            rows.append({
                "query": name,
                "closest_label": clean_result,
                "targalwayset_id": target_id,
                "result_id": result_id,
                "correct": correct,
                "time_sec_candidate_selection": elapsed_cand,
                "time_sec_disambiguation": elapsed_disamb
            })
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-{condition}-{source_name}.csv')
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")


def main():
    for source in [IMGNET_SOURCE_PATH,VGENOME_SOURCE_PATH]:
        groundtruth = load_groundtruth(VGENOME_SOURCE_PATH)
        source_name = source.split('/')[-1].split('.')[0]
        logging.info(f'Starting Experiments for {source}')
        experiment_orvis_linker(groundtruth, source_name)
        # experiment_orvis_linker_altlabel_context(groundtruth, source_name)
        #experiment_baseline(groundtruth, source_name)
        # experiment_orvis_linker_no_context(groundtruth, source_name)
        # experiment_orvis_linker_no_physical_filter(groundtruth, source_name)
        # experiment_orvis_linker_no_noun_filter(groundtruth, source_name)
        # experiment_orvis_linker_no_disambiguation(groundtruth, source_name)

if __name__ == "__main__":
    main()
