from owlready2 import *

# Load your ontology
onto = get_ontology("/home/user/pel_ws/src/orvis/orka/orvis_testing.owl").load()

# Find the Apple class
apple_class = onto.search_one(iri="*Apple")
if not apple_class:
    raise ValueError("Class 'Apple' not found in the ontology")

# Dynamically create a new class for the restriction
AppleDetector = types.new_class("AppleDetector", (Thing,))
AppleDetector.equivalent_to = [onto.canDetect.some(apple_class)]

# Run reasoning to classify individuals
with onto:
    sync_reasoner_pellet()

# Get instances of the new class
results = list(AppleDetector.instances())

# Print the results
if results:
    print("Results for DL query 'can_detect some Apple':")
    for instance in results:
        print(f" - {instance.name}")
else:
    print("No results found for the query.")
