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

# =======================
# Utility Functions
# -----------------------
def load_knowledgegraph(path_to_kg=None):
    logging.info("Loading knowledge graph...")
    g = Graph()
    g.parse(str(path_to_kg or kg_path))
    q = f"""
    SELECT ?label ?def WHERE {{
        ?c rdfs:label ?label .
        OPTIONAL {{ ?c <{definition_iri}> ?def . }}
    }}
    """
    logging.debug("Querying knowledge graph to get concepts and definition")
    results = g.query(q)
    concepts = [f"{str(r['label'])}. Definition: {str(r['def']) if r['def'] else ''}" for r in results]
    logging.debug(f"Loaded {len(concepts)} concepts. Example: {concepts[:3]}")
    return concepts


def embed_knowledgegraph(concepts, embedder):
    logging.info("Embedding knowledge graph...")
    embeddings = embedder.encode(concepts, convert_to_tensor=True)
    logging.debug(f"Embedding shape: {embeddings.shape}")
    return dict(zip(concepts, embeddings))

def embed_single_entity(string, embedder):
    logging.info(f"Embedding single entity: {string}")
    emb = embedder.encode(string, convert_to_tensor=True)
    logging.debug(f"Embedding vector: {emb[:5]}")
    return emb

def linker(embedded_entity, embedded_kg):
    logging.info("Linking entity to KG...")
    scores = {k: util.cos_sim(embedded_entity, v).item() for k, v in embedded_kg.items()}
    best_match = max(scores, key=scores.get)
    logging.debug(f"Top matches: {sorted(scores.items(), key=lambda x: -x[1])[:5]}")
    return best_match

def save_embedded_kg(embedded_kg, path="embedded_kg.npz", meta_path="embedded_kg_keys.json"):
    keys, vecs = zip(*[(k, v.cpu().numpy()) for k, v in embedded_kg.items()])
    np.savez_compressed(path, *vecs)
    json.dump(keys, open(meta_path, "w"))
    logging.info(f"Saved {len(keys)} embeddings → {path}")

def load_embedded_kg(path="embedded_kg.npz", meta_path="embedded_kg_keys.json"):
    if not Path(path).exists():
        
        return None
    keys = json.load(open(meta_path))
    arrs = np.load(path)
    
    embedded_kg = {k: tensor(arrs[f"arr_{i}"]) for i, k in enumerate(keys)}
    logging.info(f"Loaded {len(embedded_kg)} embeddings from disk.")
    return embedded_kg

# ======================
# Main function
# ----------------------
def main():
    concept_to_find = "orange has location has color orange"
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedded_kg = load_embedded_kg()
    if not embedded_kg:
        kg = load_knowledgegraph()
        embedded_kg = embed_knowledgegraph(kg, embedder)
        save_embedded_kg(embedded_kg)
    concept_emb = embed_single_entity(concept_to_find, embedder)
    closest = linker(concept_emb, embedded_kg)
    print("Closest match:", closest)

# ===================================================
# ===================================================
if __name__ == "__main__":
    main()