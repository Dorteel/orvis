
from pathlib import Path
from pathlib import Path
import owlready2
from sentence_transformers import SentenceTransformer, util
import logging, pickle
from owlready2 import get_ontology, default_world
from rdflib import Graph
import json, numpy as np
from torch import tensor

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
kg_path = Path('linking/wn_full.owl')
definition_iri = "http://example.org/wordnet.owl#definition"
identifier_iri = "http://example.org/wordnet.owl#identifier"
SOURCE_PATH = 'experiments/perceived-entity-linking/source.txt'

def load_groundtruth(source_path):
    """
    Loads the source path and prepares it for the matching
    """
    logging.info("Loading ground truth...")
    results = {}
    f = open(source_path)
    contents = f.readlines()
    f.close()
    for concept in contents:
        id = concept.split(' ')[0]
        id_corrected = id[1:] + '-n'
        names = concept.split(id)[1:][0].strip().split(', ')
        for name in names:
            results[name] = id_corrected
        
    logging.debug(f"Loaded {len(results)} concepts.")
    return results


def create_id_lookup(path_to_kg=None):
    logging.info("Loading knowledge graph...")
    g = Graph()
    g.parse(str(path_to_kg or kg_path))
    q = f"""
    SELECT ?label ?id WHERE {{
        ?c rdfs:label ?label .
        OPTIONAL {{ ?c <{identifier_iri}> ?id . }}
    }}
    """
    logging.debug("Querying knowledge graph to get concepts and identifiers")
    results = g.query(q)
    concept_id_lookup = {str(r['label']) : str(r['id']) for r in results}
    logging.debug(f"Loaded {len(concept_id_lookup)} concepts.")
    return concept_id_lookup


# =======================
# Utility Functions
# -----------------------
# def load_knowledgegraph(path_to_kg=None):
#     logging.info("Loading knowledge graph...")
#     g = Graph()
#     g.parse(str(path_to_kg or kg_path))
#     q = f"""
#     SELECT ?label ?def ?id WHERE {{
#         ?c rdfs:label ?label .
#         OPTIONAL {{ ?c <{definition_iri}> ?def . }}
#         OPTIONAL {{ ?c <{identifier_iri}> ?id . }}
#     }}
#     """
#     logging.debug("Querying knowledge graph to get concepts and definition")
#     results = g.query(q)
#     concepts_to_embed = [f"{str(r['label'])}. Definition: {str(r['def']) if r['def'] else ''}" for r in results]
#     logging.debug(f"Loaded {len(concepts_to_embed)} concepts. Example: {concepts_to_embed[:3]}")
#     return concepts_to_embed



class PerceivedEntityLinker:
    def __init__(self, kg_path='linking/wn_full.owl', model="all-MiniLM-L6-v2"):
        self.kg_path = Path(kg_path)
        self.definition_iri = "http://example.org/wordnet.owl#definition"
        self.embedder = SentenceTransformer(model)
        # Load or build KG
        if Path("embedded_kg.npz").exists():
            self.embedded_kg = self.load_embedded_kg()
        else:
            self.kg = self.load_knowledgegraph()
            self.embedded_kg = self.embed_knowledgegraph(self.kg)
            self.save_embedded_kg()

    def load_knowledgegraph(self):
        logging.info("Loading knowledge graph via SPARQL...")
        g = Graph(); g.parse(str(self.kg_path))
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
        logging.info(f"Embedding single entity: {string}")
        emb = self.embedder.encode(string, convert_to_tensor=True)
        logging.debug(f"Embedding vector (first 5 vals): {emb[:5]}")
        return emb

    def linker(self, embedded_entity):
        logging.info("Linking entity to KG...")
        scores = {k: util.cos_sim(embedded_entity, v).item() for k, v in self.embedded_kg.items()}
        best = max(scores, key=scores.get)
        logging.debug(f"Top matches: {sorted(scores.items(), key=lambda x: -x[1])[:5]}")
        return best

    def save_embedded_kg(self, path="embedded_kg.npz", meta_path="embedded_kg_keys.json"):
        keys, vecs = zip(*[(k, v.cpu().numpy()) for k, v in self.embedded_kg.items()])
        np.savez_compressed(path, *vecs)
        json.dump(keys, open(meta_path, "w"))
        logging.info(f"Saved {len(keys)} embeddings → {path}")

    def load_embedded_kg(self, path="embedded_kg.npz", meta_path="embedded_kg_keys.json"):
        if not Path(path).exists(): 
            logging.warning("No saved KG found.")
            return None
        keys = json.load(open(meta_path))
        arrs = np.load(path)
        self.embedded_kg = {k: tensor(arrs[f"arr_{i}"]) for i, k in enumerate(keys)}
        logging.info(f"Loaded {len(self.embedded_kg)} embeddings from disk.")
        return self.embedded_kg

    def find_closest(self, concept):
        if not self.embedded_kg:
            logging.error("Embedded KG not loaded or created.")
            return None
        emb = self.embed_single_entity(concept)
        return self.linker(emb)


def main():
    groundtruth = load_groundtruth(SOURCE_PATH)
    concepts_to_find = list(groundtruth.keys())
    concept_ids = create_id_lookup()
    pel = PerceivedEntityLinker()
    # Experiment part
    for concept in concepts_to_find[:5]:
        closest = pel.find_closest(concept)
        logging.debug(f"Closest match found for {concept} is:  {closest}")
        clean_result = closest.split('. Definition')[0]
        target_id = groundtruth[concept]
        result_id = concept_ids[clean_result]
        logging.debug(f"\t Target ID: {target_id}\t Result: {result_id}")

if __name__ == "__main__":
    main()