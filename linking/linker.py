from pathlib import Path
import logging, json, numpy as np, csv, time
from rdflib import Graph, Namespace, URIRef, RDFS
from torch import tensor
from sentence_transformers import SentenceTransformer, util
import os
from nltk.corpus import wordnet as wn
import Levenshtein

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

        if Path("embedded_kg.npz").exists():
            self.embedded_kg = self.load_embedded_kg()
        else:
            self.embedded_kg = self.embed_knowledgegraph(self.concepts)
            self.save_embedded_kg()
            
        logging.info("PerceivedEntityLinker initialized and ready.")

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
            # --- Type filters ---
            if self.condition == 'no-physical':
                delt_noun = 1 if self.label_to_id.get(desc)[-1] == 'n' else 0
                iri = self.label_to_iri[desc]
                mask = delt_noun          
            if self.condition == 'no-noun':
                iri = self.label_to_iri[desc]
                delt_phys = 1 if str(iri) in self.physical_descendants else 0
                mask =  delt_phys     
            else:
                delt_noun = 1 if self.label_to_id.get(desc)[-1] == 'n' else 0
                iri = self.label_to_iri[desc]
                delt_phys = 1 if str(iri) in self.physical_descendants else 0
                mask = delt_noun * delt_phys

            if self.condition != 'no-context':
                # --- Hierarchical context ---
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
    def __init__(self, kg_path='linking/wn_full.owl', model="all-MiniLM-L6-v2"):
        self.kg_path = Path(kg_path)
        self.definition_iri = "http://example.org/wordnet.owl#definition"
        self.concepts, self.kg, self.id_to_label, self.label_to_iri, self.iri_to_id = self.load_knowledgegraph()
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.target_concept = None
        logging.info("BaseLine PerceivedEntityLinker initialized and ready.")

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
        iri_id_lookup = {
            str(r['c']): f"{r['id']}"
            for r in results
            if r['id'] is not None
        }
        concept_iri_lookup = {
            f"{r['label']}. Definition: {r['def'] if r['def'] else ''}": str(r['c'])
            for r in results
        }
        concepts = list(concept_iri_lookup.keys())
        logging.debug(f"Loaded {len(concepts)} concepts. Example: {concepts[:3]}")
        return concepts, g, concept_id_lookup, concept_iri_lookup, iri_id_lookup

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

    def get_entities_by_label(self, label):
        logging.debug(f'...Getting entities by label: {label}')
        q = f"""
        SELECT ?entity WHERE {{
            ?entity rdfs:label ?l .
            FILTER (lcase(str(?l)) = lcase("{label}"))
        }}
        """
        results = list(self.kg.query(q))
        entities = [str(row["entity"]) for row in results]
        logging.debug(f'...Found entities: {entities}')
        return entities

    def get_synset(self, entity_name):

        # Normalize entity name (WordNet is case-sensitive and expects underscores)
        query = entity_name.strip().replace(" ", "_").lower()
        synsets = wn.synsets(query)
        if not synsets:
            logging.debug(f"No synsets found for entity: {entity_name}")
            return []
        logging.debug(f"Found {len(synsets)} synsets for '{entity_name}': {[s.name() for s in synsets]}")
        return synsets

    def get_names(self, synset):
        if not synset:
            return []
        names = synset.lemma_names()
        logging.debug(f"Lemmas for {synset.name()}: {names}")
        return names


    def get_synset_similarity(self, synsets_A, synset_B):
        if not synsets_A or not synset_B:
            return 0.0

        target_name = synset_B.name().split(".")[0].replace("_", " ").lower()
        max_score = 0.0

        for sA in synsets_A:
            nameA = sA.name().split(".")[0].replace("_", " ").lower()
            # Levenshtein ratio gives normalized similarity (1 = identical)
            score = Levenshtein.ratio(nameA, target_name)
            max_score = max(max_score, score)

        return max_score

    def get_class(self, entity_iri):
        q = f"""
        SELECT ?class_label WHERE {{
            <{entity_iri}> rdfs:subClassOf ?cls .
            OPTIONAL {{ ?cls rdfs:label ?class_label . }}
        }}
        """
        results = list(self.kg.query(q))

        class_labels = []
        for row in results:
            if row.get("class_label"):
                class_labels.append(str(row["class_label"]))
            else:
                # fallback to the class IRI if label is missing
                q2 = f"""
                SELECT ?cls WHERE {{
                    <{entity_iri}> rdf:type ?cls .
                }}
                """
                class_labels.extend([str(r["cls"]) for r in self.kg.query(q2)])

        return class_labels

    def candidate_selection(self, concept, k=5):
        logging.debug(f"Selecting candidates for: {concept}")
        self.target_concept = concept
        start = time.time()
        # --- 1. Retrieve WordNet synsets for the observed entity ---
        synsets = self.get_synset(concept)
        if not synsets:
            logging.debug("No synsets found, returning the entity itself.")
            elapsed = round(time.time() - start, 2)
            return [(concept, 1.0)], elapsed

        candidates = []

        # --- 2. Iterate over each synset ---
        for sn in synsets:
            # Retrieve all names (lemmas) associated with the synset
            raw_names = self.get_names(sn)            
            # Add variations:
            names = set(self.get_altlabels_by_label(concept) + raw_names)
            logging.debug(f"Synset {sn} → {names}")
            # --- 3. Compute similarity between the perceived entity and each name ---
            for name in names:
                sim_score = Levenshtein.ratio(concept, name)
                candidates.append((name, sim_score))
                logging.debug(f"Similarity({concept}, {name}) = {sim_score:.3f}")

        # --- 4. Sort candidates by descending similarity score ---
        candidates.sort(key=lambda x: x[1], reverse=True)
        logging.debug(f"Top candidate labels: {candidates[:k]}")

        candidate_entities = []
        for candidate, _ in candidates[:k]:
            results = self.get_entities_by_label(candidate)
            candidate_entities.extend(results)
            logging.debug(f"Candidate entities found for {candidate} candidates: {results}")
        elapsed = round(time.time() - start, 2)
        return candidate_entities[:k], elapsed

    def disambiguate(self, candidates):

        logging.debug(f"Disambiguating entity: {self.target_concept}")
        start = time.time()
        # --- 1. Get synsets of the observed entity (WordNet) ---
        target_ss = self.get_synset(self.target_concept)
        if not target_ss:
            logging.debug(f"No synsets found for observed entity '{self.target_concept}', returning candidates as-is.")
            return [(c, 0.0) for c in candidates]

        scores = []

        # --- 2. Iterate through each candidate ---
        for c in candidates:
            logging.debug(f"Examining Candidate '{c}'")
            high_score = 0.0

            # --- 3. Retrieve candidate's ontology classes from target KG ---
            candidate_classes = self.get_class(c)
            logging.debug(f"Candidate '{c}' has classes: {candidate_classes}")

            # --- 4. Compare the synsets of each class to the observed entity synsets ---
            for cls in candidate_classes:
                class_ss = self.get_synset(cls)

                for ss in class_ss:
                    # Compute maximum pairwise similarity between the synsets
                    sim_score = self.get_synset_similarity(target_ss, ss)

                    if sim_score > high_score:
                        high_score = sim_score

            scores.append((c, high_score))
            logging.debug(f"Candidate '{c}' → best synset similarity = {high_score:.3f}")

        # --- 5. Sort candidates by descending similarity ---
        scores.sort(key=lambda x: x[0], reverse=True)
        logging.debug(f"Top disambiguation results: {scores[:5]}")
        elapsed = round(time.time() - start, 2)
        return scores, elapsed  

# ==================
# EXPERIMENTS
# ------------------

def experiment_baseline(groundtruth, source_name):
    
    linker = BaseLinePerceivedEntityLinker()

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
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-baseline-{source_name}.csv')
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")

def experiment_orvis_linker(groundtruth, source_name):
    linker = PerceivedEntityLinker()
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
    
    RESULTS_PATH = Path(f'experiments/perceived-entity-linking/results-complete-{source_name}.csv')
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

def main():
    for source in [IMGNET_SOURCE_PATH, VGENOME_SOURCE_PATH]:
        groundtruth = load_groundtruth(source)
        source_name = source.split('/')[-1].split('.')[0]
        experiment_orvis_linker(groundtruth, source_name)
        experiment_baseline(groundtruth, source_name)
        experiment_orvis_linker_no_context(groundtruth, source_name)
        experiment_orvis_linker_no_physical_filter(groundtruth, source_name)
        experiment_orvis_linker_no_noun_filter(groundtruth, source_name)
        experiment_orvis_linker_no_disambiguation(groundtruth, source_name)

if __name__ == "__main__":
    main()
