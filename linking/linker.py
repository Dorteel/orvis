from pathlib import Path
import logging, json, numpy as np, csv, time
from rdflib import Graph
from torch import tensor
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
kg_path = Path('linking/wn_full.owl')
definition_iri = "http://example.org/wordnet.owl#definition"
identifier_iri = "http://example.org/wordnet.owl#identifier"
SOURCE_PATH = 'experiments/perceived-entity-linking/source.txt'
RESULTS_PATH = Path('experiments/perceived-entity-linking/results-v2.csv')


def load_groundtruth(source_path):
    logging.info("Loading ground truth...")
    results = {}
    with open(source_path) as f:
        for concept in f:
            id_ = concept.split(' ')[0]
            id_corrected = id_[1:] + '-n'
            names = concept.split(id_)[1:][0].strip().split(', ')
            results[id_corrected] = names
    logging.debug(f"Loaded {len(results)} concepts.")
    return results


def create_id_lookup(path_to_kg=None):
    logging.info("Creating ID lookup from KG...")
    g = Graph()
    g.parse(str(path_to_kg or kg_path))
    q = f"""
    SELECT ?label ?id ?def WHERE {{
        ?c rdfs:label ?label .
        OPTIONAL {{ ?c <{identifier_iri}> ?id . }}
        OPTIONAL {{ ?c <{definition_iri}> ?def . }}
    }}
    """
    results = g.query(q)
    concept_id_lookup = {str(r['id']) : f"{str(r['label'])}. Definition: {str(r['def']) if r['def'] else ''}" for r in results}
    logging.debug(f"Loaded {len(concept_id_lookup)} concepts.")
    return concept_id_lookup


class PerceivedEntityLinker:
    def __init__(self, kg_path='linking/wn_full.owl', model="all-MiniLM-L6-v2"):
        self.kg_path = Path(kg_path)
        self.definition_iri = "http://example.org/wordnet.owl#definition"
        self.embedder = SentenceTransformer(model)
        if Path("embedded_kg.npz").exists():
            self.embedded_kg = self.load_embedded_kg()
        else:
            self.kg = self.load_knowledgegraph()
            self.embedded_kg = self.embed_knowledgegraph(self.kg)
            self.save_embedded_kg()
        logging.info("PerceivedEntityLinker initialized and ready.")

    def load_knowledgegraph(self):
        logging.info("Loading knowledge graph via SPARQL...")
        g = Graph()
        g.parse(str(self.kg_path))
        q = f"""
        SELECT ?label ?def WHERE {{
            ?c rdfs:label ?label .
            OPTIONAL {{ ?c <{self.definition_iri}> ?def . }}
        }}
        """
        results = g.query(q)
        concepts = [f"{str(r['label'])}. Definition: {str(r['def']) if r['def'] else ''}" for r in results]
        logging.debug(f"Loaded {len(concepts)} concepts. Example: {concepts[:3]}")
        return concepts

    def embed_knowledgegraph(self, concepts):
        logging.info("Embedding knowledge graph...")
        embeddings = self.embedder.encode(concepts, convert_to_tensor=True)
        self.embedded_kg = dict(zip(concepts, embeddings))
        logging.debug(f"Embedding shape: {embeddings.shape}")
        return self.embedded_kg

    def embed_single_entity(self, string):
        logging.info(f"Embedding entity: {string}")
        emb = self.embedder.encode(string, convert_to_tensor=True)
        return emb

    def linker(self, embedded_entity):
        scores = {k: util.cos_sim(embedded_entity, v).item() for k, v in self.embedded_kg.items()}
        best = max(scores, key=scores.get)
        return best

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

    def find_closest(self, concept):
        start = time.time()
        emb = self.embed_single_entity(concept)
        closest = self.linker(emb)
        elapsed = round(time.time() - start, 2)
        logging.debug(f"Closest match for '{concept}' → '{closest}' (time: {elapsed}s)")
        return closest, elapsed


def main():
    groundtruth = load_groundtruth(SOURCE_PATH)
    id_to_label = create_id_lookup()
    label_to_id = {v: k for k, v in id_to_label.items()}  # reverse map: label → id
    pel = PerceivedEntityLinker()

    rows = []
    for i, target_id in enumerate(list(groundtruth.keys())):
        names = groundtruth[target_id]
        for name in names:
            closest, elapsed = pel.find_closest(name)
            clean_result = closest.split('. Definition')[0].strip()
            result_id = label_to_id.get(closest, 'N/A')
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
                "time_sec": elapsed
            })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved → {RESULTS_PATH}")

if __name__ == "__main__":
    main()
