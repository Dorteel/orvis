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
import faiss


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
# Base Linker (shared logic)
# =============================

class BaseLinker:
    def __init__(self, kg_path, model):
        logging.debug(f"[INIT] Loading model '{model}' and KG '{kg_path}'")
        self.kg_path = Path(kg_path)
        self.definition_iri = "http://example.org/wordnet.owl#definition"
        self.altlabel_iri = "http://example.org/wordnet.owl#altLabel"

        # Model
        self.embedder = SentenceTransformer(model)
        logging.debug("[INIT] SentenceTransformer loaded.")

        # KG + maps
        self.concepts, self.kg, self.id_to_label, self.label_to_iri = self.load_knowledgegraph()
        logging.debug(f"[INIT] Loaded KG: {len(self.concepts)} concepts.")

        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        logging.debug(f"[INIT] Built label_to_id map: {len(self.label_to_id)} entries.")

        # --- precompute graph structures ---
        self.ALT = URIRef(self.altlabel_iri)
        self.DEF = URIRef(self.definition_iri)
        self.IDENT = URIRef(identifier_iri)
        self.iri_to_id: dict[str, str] = {}
        self.altlabels = {}
        self.parents = {}
        self.iri_to_label = {}
        self.iri_to_def = {}

        logging.debug("[INIT] Scanning KG triples...")
        for s, p, o in self.kg.triples((None, None, None)):
            s_str = str(s)
            if p == RDFS.label:
                self.iri_to_label[s_str] = str(o)
            elif p == self.ALT:
                self.altlabels.setdefault(s_str, set()).add(str(o))
            elif p == RDFS.subClassOf:
                self.parents.setdefault(s_str, set()).add(str(o))
            elif p == self.DEF:
                self.iri_to_def[s_str] = str(o)
            elif p == self.IDENT:
                self.iri_to_id[s_str] = str(o)

        logging.debug(f"[INIT] Triples processed: labels={len(self.iri_to_label)}, altlabels={sum(len(v) for v in self.altlabels.values())}, parents={len(self.parents)}")

        # Physical entity index
        self.physical_descendants = self.get_descendants(EX["C_00001930-n"])
        logging.debug(f"[INIT] Physical descendants count: {len(self.physical_descendants)}")

        # Embeddings
        if Path("embedded_kg.npz").exists():
            logging.debug("[INIT] Loading cached KG embeddings...")
            self.embedded_kg = self.load_embedded_kg()
        else:
            logging.debug("[INIT] Embedding KG concepts (first run)...")
            self.embedded_kg = self.embed_knowledgegraph(self.concepts)
            self.save_embedded_kg()
        logging.debug(f"[INIT] Loaded {len(self.embedded_kg)} KG embeddings.")

        # Build FAISS index (L2 normalized embeddings for cosine search)
        logging.debug("[FAISS] Preparing embedding matrix for FAISS...")
        vecs = [v.cpu().numpy().astype("float32") for v in self.embedded_kg.values()]
        self.kg_matrix = np.vstack(vecs)

        # Normalize for cosine similarity -> L2 distance equivalence
        faiss.normalize_L2(self.kg_matrix)

        dim = self.kg_matrix.shape[1]  # embedding dimension

        logging.debug(f"[FAISS] Building index of size {len(self.kg_matrix)} x {dim}")
        self.faiss_index = faiss.IndexFlatIP(dim)  # inner product ≡ cosine for normalized vectors

        # add vectors to index
        self.faiss_index.add(self.kg_matrix)

        # maintain mapping from FAISS row → concept string
        self.faiss_keys = list(self.embedded_kg.keys())

        logging.info(f"[FAISS] Index built with {self.faiss_index.ntotal} vectors.")

        logging.info(f"{self.__class__.__name__} initialized")

    def get_descendants(self, root):
        logging.debug(f"[DESC] Computing descendants for {root}")
        q = f"SELECT ?x WHERE {{ ?x rdfs:subClassOf* <{root}> }}"
        res = {str(r[0]) for r in self.kg.query(q)}
        logging.debug(f"[DESC] {len(res)} descendants found")
        return res
    
    # --- KG Load ---
    def load_knowledgegraph(self):
        logging.debug(f"[KG] Parsing KG from {self.kg_path}")
        g = Graph()
        g.parse(str(self.kg_path))
        logging.debug("[KG] KG parsed. Running label/id/definition query...")

        q = f"""
        SELECT ?label ?id ?def ?c WHERE {{
            ?c rdfs:label ?label .
            OPTIONAL {{ ?c <{identifier_iri}> ?id . }}
            OPTIONAL {{ ?c <{definition_iri}> ?def . }}
        }}
        """
        rows = list(g.query(q))
        logging.debug(f"[KG] Query returned {len(rows)} rows.")

        id_map = {str(r['id']): f"{r['label']}. Definition: {r['def'] or ''}"
                  for r in rows if r['id']}
        iri_map = {f"{r['label']}. Definition: {r['def'] or ''}": str(r['c'])
                   for r in rows}

        logging.debug(f"[KG] Mapped {len(id_map)} ids and {len(iri_map)} labels.")
        return list(iri_map.keys()), g, id_map, iri_map

    def load_embedded_kg(self, path="embedded_kg.npz", meta="embedded_kg_keys.json"):
        logging.debug(f"[EMB] Loading cached embeddings from {path}")
        keys = json.load(open(meta))
        arrs = np.load(path)
        emb = {k: tensor(arrs[f"arr_{i}"]) for i, k in enumerate(keys)}
        logging.debug(f"[EMB] Loaded {len(emb)} cached embeddings.")
        return emb

    @lru_cache(maxsize=None)
    def get_hierarchy_cached(self, iri: str, depth: int = 2) -> str:
        logging.debug(f"[CTX] Building hierarchy for {iri} depth={depth}")
        context = []
        current = {iri}
        visited = set()

        for _ in range(depth):
            nxt = set()
            for e in current:
                if e in visited:
                    continue
                visited.add(e)

                l = self.iri_to_label.get(e, "")
                d = self.iri_to_def.get(e, "")
                if l or d:
                    context.append(f"{l}. {d}")

                for p in self.parents.get(e, ()):
                    nxt.add(p)
            current = nxt

        ctx_str = " ".join(context)
        logging.debug(f"[CTX] Context built: '{ctx_str[:120]}...'")
        return ctx_str

    def get_hierarchy(self, iri, depth=2):
        return self.get_hierarchy_cached(iri, depth)

    def get_altlabels(self, label: str):
        iri = self.label_to_iri.get(label.lower())
        alts = sorted(self.altlabels.get(iri, set()) | {label}) if iri else [label]
        logging.debug(f"[ALT] {label} -> {alts}")
        return alts

    def embed_knowledgegraph(self, concepts):
        logging.debug("[EMB] Embedding KG...")
        embs = self.embedder.encode(concepts, convert_to_tensor=True)
        logging.debug("[EMB] Done.")
        return dict(zip(concepts, embs))

    def save_embedded_kg(self, path="embedded_kg.npz", meta="embedded_kg_keys.json"):
        logging.debug(f"[EMB] Saving embeddings to {path}")
        keys, vecs = zip(*[(k, v.cpu().numpy()) for k, v in self.embedded_kg.items()])
        np.savez_compressed(path, *vecs)
        json.dump(keys, open(meta, "w"))
        logging.debug("[EMB] Saved cached embeddings.")

    @lru_cache(maxsize=None)
    def embed_cached(self, text: str):
        logging.debug(f"[EMB-Q] Encoding text: '{text[:80]}'")
        return self.embedder.encode(text, convert_to_tensor=True)

    def embed(self, text):
        return self.embed_cached(text)

    def knn(self, q_emb, k=5):
        # encode query → numpy float32
        q = q_emb.cpu().numpy().astype("float32")
        q = q.reshape(1, -1)

        # normalize for cosine similarity
        faiss.normalize_L2(q)

        # FAISS search (returns scores and indices)
        scores, idxs = self.faiss_index.search(q, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            key = self.faiss_keys[idx]          # concept label string
            results.append((key, float(score))) # consistent return format
        
        logging.debug(f"[FAISS-KNN] Query top: {results[0] if results else None}")
        return results

# =================================
#  PEL
# ---------------------------------


class PerceivedEntityLinker(BaseLinker):
    def __init__(self, kg_path='linking/wn_full.owl', model="all-MiniLM-L6-v2"):
        super().__init__(kg_path, model)

    def candidate_selection(self, concept, k=5):
        start = time.time()
        self.target_emb = self.embed(concept)
        return self.knn(self.target_emb, k), round(time.time() - start, 4)

    def disambiguate(self, candidates, depth=2):
        results = []
        start = time.time()
        for label, _ in candidates:
            iri = self.label_to_iri[label]
            noun = self.label_to_id.get(label, '').endswith('n')
            phys = iri in self.physical_descendants
            mask = int(noun and phys)
            ctx = f"{label} {self.get_hierarchy(iri, depth)}"
            score = util.cos_sim(self.target_emb, self.embed(ctx)).item()
            logging.debug(f"[DISAMB] Evaluating {label}, mask={mask}, score={score:.4f}")
            results.append((label, score * mask))
        return sorted(results, key=lambda x: x[1], reverse=True), round(time.time() - start, 4)

# =================================
#  Ablated PEL
# ---------------------------------

class AblatedPerceivedEntityLinker(BaseLinker):
    def __init__(self, condition, kg_path='linking/wn_full.owl', model="all-MiniLM-L6-v2"):
        self.condition = condition
        super().__init__(kg_path, model)

    def _mask(self, label, iri):
        noun = self.label_to_id.get(label, '').endswith('n')
        phys = iri in self.physical_descendants

        if self.condition == 'no-physical': return int(noun)
        if self.condition == 'no-noun':     return int(phys)
        return int(noun and phys)

    def candidate_selection(self, concept, k=5):
        start = time.time()
        self.target_emb = self.embed(concept)
        return self.knn(self.target_emb, k), round(time.time() - start, 4)

    def disambiguate(self, candidates, depth=2):
        results = []
        start = time.time()
        for label, base_score in candidates:

            iri = self.label_to_iri[label]
            mask = self._mask(label, iri)

            if self.condition == 'no-context':
                score = base_score
            else:
                ctx = label + " " + self.get_hierarchy(iri, depth)
                score = util.cos_sim(self.target_emb, self.embed(ctx)).item()
            logging.debug(f"[DISAMB-{self.condition}] {label}, base={base_score:.4f}, mask={mask}, score={score:.4f}")
            results.append((label, mask * score))
        return sorted(results, key=lambda x: x[1], reverse=True), round(time.time() - start, 4)

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

    def get_entities_by_label(self, label):
        # Resolve label directly to entity IRI if present; keep API compatibility (list).
        iri = self.label_to_iri.get(label.lower())
        return [iri] if iri else []

    def get_class(self, entity_iri):
        # Return direct rdfs:subClassOf parents (as IRIs); empty list if none.
        return list(self.subclasses.get(entity_iri, []))
    
    def get_altlabels(self, label: str):
        iri = self.label_to_iri.get(label.lower())
        if not iri:
            return [label]
        return sorted(self.altlabels.get(iri, set()) | {label})
    
    def candidate_selection(self, concept, k=5):
        # Build candidate entity IRIs via altLabels+lemmas; score by RapidFuzz and pick top-k.
        start = time.time()
        self.target_concept = concept
        synsets = self.get_synset(concept)
        if not synsets:
            return [(concept, 1.0)], round(time.time() - start, 2)

        candidates = []
        altlabels = self.get_altlabels(concept)
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

def run_experiment(linker, groundtruth, source_name, exp_name, skip_disamb=False):
    rows = []

    for i, target_id in enumerate(list(groundtruth.keys())):
        for name in groundtruth[target_id]:
            # candidate selection
            closests, t_cand = linker.candidate_selection(name)

            # disambiguation
            if skip_disamb:
                closests_ordered, t_disamb = closests, 0.0
            else:
                closests_ordered, t_disamb = linker.disambiguate(closests)

            # ranked KG IDs
            ranked_ids = []
            for label, _ in closests_ordered:
                iri = linker.label_to_iri.get(label)
                candidate_id = linker.iri_to_id.get(iri) if iri else None
                ranked_ids.append(candidate_id)

            # compute rank
            rank = None
            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1  # 1-based index
            else:
                rank = -1

            # hits@k booleans
            hit1 = (rank == 1)
            hit3 = (rank != -1 and rank <= 3)
            hit5 = (rank != -1 and rank <= 5)

            # top prediction fields
            if ranked_ids:
                best_id = ranked_ids[0]
                best_label = closests_ordered[0][0].split('. Definition')[0]
            else:
                best_id, best_label = 'N/A', 'N/A'

            rows.append({
                "query": name,
                "closest_label": best_label,
                "target_id": target_id,
                "result_id": best_id,
                "correct": (best_id == target_id),
                "rank": rank,
                "hit@1": hit1,
                "hit@3": hit3,
                "hit@5": hit5,
                "time_sec_candidate_selection": t_cand,
                "time_sec_disambiguation": t_disamb
            })

    # save CSV
    out = Path(f"experiments/perceived-entity-linking/results/{exp_name}-{source_name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    logging.info(f"Results saved → {out}")


EXPERIMENTS = {
    #"baseline":          (lambda: BaseLinePerceivedEntityLinker(), False),
    "complete":          (lambda: PerceivedEntityLinker(), False),
    "no-context":        (lambda: AblatedPerceivedEntityLinker("no-context"), False),
    "no-physical":       (lambda: AblatedPerceivedEntityLinker("no-physical"), False),
    "no-noun":           (lambda: AblatedPerceivedEntityLinker("no-noun"), False),
    "no-disamb":         (lambda: PerceivedEntityLinker(), True),
    "altlabel-context":  (lambda: AblatedPerceivedEntityLinker("altlabel-context"), False),
}


def main():
    for source in [ VGENOME_SOURCE_PATH]:
        groundtruth = load_groundtruth(source)
        source_name = Path(source).stem

        for name, (make_linker, skip_disamb) in EXPERIMENTS.items():
            logging.info(f"Running {name} on {source_name}")
            linker = make_linker()
            run_experiment(linker, groundtruth, source_name, name, skip_disamb=skip_disamb)


if __name__ == "__main__":
    main()
