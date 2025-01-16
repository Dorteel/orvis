import rospy
import actionlib
# from some_msgs.msg import PickupAction, PickupGoal
# from some_msgs.srv import AnnotatorService, AnnotatorServiceRequest
import random
import os
from owlready2 import get_ontology, default_world, sync_reasoner_pellet

def query_annotators(obs_graph, object):
    """
    Queries the observation graph for annotators able to detect the given object using a SPARQL query.
    Adds a simulated entity of the given type to the ontology, performs reasoning, and then executes the query.
    
    :param obs_graph: The loaded ontology graph (owlready2 ontology object).
    :param object: The target object (assumed to be an IRI or identifier).
    :return: List of results from the SPARQL query, or None if no results are found.
    """
    try:
        rospy.loginfo(f"Adding a simulated entity of type {object} to the ontology...")
        
        # Step 1: Add a simulated entity of type 'object' to the ontology
        with obs_graph:
            simulated_entity = obs_graph[object](f"SimulatedEntity_{object}")
            # simulated_entity.is_a.append()  # Assign type `object`
            rospy.loginfo("Running reasoning...")
            sync_reasoner_pellet(infer_property_values=True, debug=0)
            rospy.loginfo("Reasoning complete.")

        # Step 2: Construct and run the SPARQL query
        rospy.loginfo(f"Querying the observation graph for annotators capable of detecting {object}...")
        sparql_query_annotators = f"""
        PREFIX sosa: <http://www.w3.org/ns/sosa/>
        PREFIX ssn: <http://www.w3.org/ns/ssn/>
        PREFIX orka: <https://w3id.org/def/orka#>
        PREFIX oboe: <http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#>

        SELECT DISTINCT ?annotator
        WHERE {{
          # Query for the simulated entity and related annotators
          ?temp_entity a orka:{object} .
          ?annotator orka:canDetect ?temp_entity .
        }}
        """
        results = list(default_world.sparql(sparql_query_annotators, error_on_undefined_entities=False))
        rospy.loginfo(f"SPARQL query returned {len(results)} results.")
        print(results)
        return results if results else None
    
    except Exception as e:
        rospy.logerr(f"Error running query_annotators function: {e}")
        return None

def get_obs_graph():
    """
    Fetches and loads the most recently modified observation graph (.owl file) 
    from the knowledge base directory using owlready2.
    Returns the loaded ontology or None if no .owl file exists.
    """
    rospy.loginfo("Fetching observation graph...")

    # Path to the obs_graphs directory (one level up from the script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    obs_graph_dir = os.path.join(os.path.dirname(script_dir), "obs_graphs")

    # Check if the directory exists
    if not os.path.exists(obs_graph_dir):
        rospy.logwarn(f"The directory {obs_graph_dir} does not exist.")
        return None

    # Get all .owl files in the directory
    owl_files = [os.path.join(obs_graph_dir, f) for f in os.listdir(obs_graph_dir) if f.endswith(".owl")]

    # If no .owl files are found, return None
    if not owl_files:
        rospy.logwarn(f"No .owl files found in {obs_graph_dir}.")
        return None

    # Find the most recently modified .owl file
    latest_obs_graph_path = max(owl_files, key=os.path.getmtime)
    rospy.loginfo(f"Latest observation graph found: {latest_obs_graph_path}")

    # Load the ontology using owlready2
    try:
        ontology = get_ontology(latest_obs_graph_path).load()
        rospy.loginfo(f"Ontology successfully loaded from {latest_obs_graph_path}.")
        return ontology
    except Exception as e:
        rospy.logerr(f"Failed to load ontology: {e}")
        return None

def create_obs_graph():
    """
    Creates a new observation graph.
    Returns the newly created observation graph.
    """
    rospy.loginfo("Creating a new observation graph...")
    return {"graph": "new_obs_graph"}


def query_location(obs_graph, object):
    """
    Queries the observation graph for the location of the given object using a SPARQL query.
    :param obs_graph: The loaded ontology graph (owlready2 ontology object).
    :param object: The target object (assumed to be an IRI or identifier).
    :return: List of results from the SPARQL query, or None if no results are found.
    """
    rospy.loginfo(f"Querying the observation graph for the location of {object}...")

    sparql_query_location = f"""
    PREFIX sosa: <http://www.w3.org/ns/sosa/>
    PREFIX ssn: <http://www.w3.org/ns/ssn/>
    PREFIX orka: <https://w3id.org/def/orka#>
    PREFIX oboe: <http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#>

    SELECT ?loc ?ent WHERE {{
      ?ent a orka:{object} .
      ?loc_instance a orka:Location .
      ?loc_instance oboe:characteristicFor ?ent .
      ?loc_instance orka:hasValue ?loc .                               
    }}
    """

    try:
        # Run the SPARQL query on the ontology
        results = list(default_world.sparql(sparql_query_location, error_on_undefined_entities=False))
        rospy.loginfo(f"SPARQL query returned {len(results)} results.")
        return results if results else None
    except Exception as e:
        rospy.logerr(f"Error running SPARQL query: {e}")
        return None

def call_annotator(annotator, object):
    """
    Calls the annotator service to detect the given object.
    """
    rospy.loginfo(f"Calling annotator service '{annotator}' to detect {object}...")
    try:
        rospy.wait_for_service(annotator, timeout=5.0)
        annotator_service = rospy.ServiceProxy(annotator, AnnotatorService)
        request = AnnotatorServiceRequest()
        request.object = object  # Populate request fields as required
        response = annotator_service(request)
        rospy.loginfo(f"Annotator service '{annotator}' response: {response}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to call annotator service '{annotator}': {e}")


def pickup_object(object_position):
    """
    Sends a goal to the pickup action server to pick up an object at the given position.
    """
    rospy.loginfo(f"Sending goal to pickup action for position {object_position}...")

    # Create an action client for the pickup action
    client = actionlib.SimpleActionClient('/pickup_action', PickupAction)

    # Wait for the action server to be available
    rospy.loginfo("Waiting for pickup action server...")
    client.wait_for_server()

    # Create and send the goal
    goal = PickupGoal()
    goal.position.x = object_position["x"]
    goal.position.y = object_position["y"]

    rospy.loginfo(f"Sending pickup goal: {goal}")
    client.send_goal(goal)

    # Wait for the result
    client.wait_for_result()
    result = client.get_result()

    rospy.loginfo(f"Pickup action completed with result: {result}")


# Main script
if __name__ == "__main__":
    rospy.init_node("object_locator_node")

    #fruit_salad_items = ['Banana', 'Apple', 'Strawberry', 'Orange', 'Pineapple']
    fruit_salad_items = ['Banana']


    for fruit in fruit_salad_items:
        rospy.loginfo(f"Processing {fruit}...")
        fruit_position = None

        obs_graph = get_obs_graph()
        if not obs_graph:
            obs_graph = create_obs_graph()

        options_left = True

        fruit_position = query_location(obs_graph, fruit)
        while options_left and not fruit_position:
            capable_annotators = query_annotators(obs_graph, fruit)
            for annotator in capable_annotators:
                call_annotator(annotator, fruit)
                obs_graph = get_obs_graph()
                fruit_position = query_location(obs_graph, fruit)

                if fruit_position:
                    break

                capable_annotators.remove(annotator)
                if not capable_annotators:
                    options_left = False

        if fruit_position:
            pickup_object(fruit_position)
        else:
            rospy.loginfo(f"{fruit} not found!")
