#!/home/user/pel_ws/pel_venv/bin/python

# This file takes the model configurations, and adds each available model to ORKA

import rospy
import os
import yaml
import types

from xml.etree import ElementTree as ET

from owlready2 import get_ontology, sync_reasoner_pellet, Imp
from orvis.srv import LoadModels, LoadModelsResponse  # Updated service definition

# Constants for paths
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_PATH, "orvis/config/models")
ONTOLOGY_PATH = os.path.join(BASE_PATH, "orvis/orka/orka.owl")


def fix_ontology_imports(ontology_path, fixed_ontology_path):
    """
    Fixes <owl:imports> lines in the ontology to include absolute paths.

    Args:
        ontology_path (str): Path to the original ontology file.
        fixed_ontology_path (str): Path to save the fixed ontology file.

    Returns:
        str: Status message indicating success or failure.
    """
    try:
        # Parse the ontology file as XML
        tree = ET.parse(ontology_path)
        root = tree.getroot()

        # Namespace mapping for RDF and OWL
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'owl': 'http://www.w3.org/2002/07/owl#'
        }

        # Iterate through all <owl:imports> elements and fix paths
        for owl_import in root.findall('.//owl:imports', namespaces):
            import_path = owl_import.get(f'{{{namespaces["rdf"]}}}resource')

            if import_path.startswith('file://'):
                # Extract the relative path from the file URI
                relative_path = import_path.replace('file://', '')
                absolute_path = os.path.abspath(os.path.join(os.path.dirname(ontology_path), relative_path))

                # Update the resource attribute with the absolute path
                owl_import.set(f'{{{namespaces["rdf"]}}}resource', f'file://{absolute_path}')
                print(f"Fixed import: {relative_path} -> {absolute_path}")

        # Save the fixed ontology file
        tree.write(fixed_ontology_path, encoding='utf-8', xml_declaration=True)
        return f"Ontology fixed and saved at {fixed_ontology_path}"
    except Exception as e:
        return f"Error fixing ontology imports: {e}"


def load_models(updated_ontology_name):
    """
    Loads all model configurations from the specified path into the ontology.

    Args:
        updated_ontology_name (str): Name of the updated ontology file to save.

    Returns:
        str: Status message indicating the result of the operation.
    """
    if not os.path.isdir(MODEL_PATH):
        print(BASE_PATH)
        return f"Model path does not exist: {MODEL_PATH}"

    fixed_ontology_path = os.path.join(os.path.dirname(ONTOLOGY_PATH), "fixed_orka.owl")
    fix_status = fix_ontology_imports(ONTOLOGY_PATH, fixed_ontology_path)

    if "Error" in fix_status:
        rospy.logerr(fix_status)
        return fix_status


    if not os.path.isfile(fixed_ontology_path):
        return f"Ontology path does not exist: {fixed_ontology_path}"

    # Load the ontology
    try:
        ontology = get_ontology(fixed_ontology_path).load()
        rospy.loginfo("Ontology loaded successfully.")
    except Exception as e:
        return f"Failed to load ontology: {e}"

    added_annotators = []

    # Iterate through each file in the models directory
    for filename in os.listdir(MODEL_PATH):
        if filename.endswith(".yaml"):
            filepath = os.path.join(MODEL_PATH, filename)

            # Load the YAML configuration
            try:
                with open(filepath, 'r') as file:
                    config = yaml.safe_load(file)
            except Exception as e:
                rospy.logerr(f"Error reading file {filename}: {e}")
                continue

            # Extract model details
            annotator_name = config.get('annotator', {}).get('name', 'UnknownAnnotator')
            task_type = config.get('annotator', {}).get('task_type', 'UnknownTask')
            service_name = f"/annotators/{task_type}/{annotator_name}/detect"
            detected_property = config.get('annotator', {}).get('detected_property', [])
            annotator_class_name = config.get('annotator', {}).get('type', 'UnknownType')
            # model_class = config.get('imports', {}).get('model_class', 'UnknownModelClass')
            # processor_class = config.get('imports', {}).get('processor_class', 'UnknownProcessorClass')
            detected_labels = config.get('detection', {}).get('labels', [])
            algorithm_type = task_type + 'Algorithm'

            # Add a new annotator to the ontology
            try:
                # Ensure the algorithm class exists
                if not hasattr(ontology, algorithm_type)  or getattr(ontology, algorithm_type) is None:
                    rospy.loginfo(f"Creating subclass '{algorithm_type}' as a ComputerVisionAlgorithm.")
                    with ontology:
                        algorithm_class = types.new_class(algorithm_type, (ontology.ComputerVisionAlgorithm,))
                else:
                    algorithm_class = getattr(ontology, algorithm_type)
                    rospy.loginfo(f"Class {algorithm_type} found: {algorithm_class}")

                # Ensure the annotator-specific class exists
                if not hasattr(ontology, annotator_class_name) or getattr(ontology, annotator_class_name) is None:
                    rospy.loginfo(f"Creating subclass '{annotator_class_name}' for algorithm type '{algorithm_type}'.")
                    with ontology:
                        annotator_class = types.new_class(annotator_class_name, (algorithm_class,))
                else:
                    annotator_class = getattr(ontology, annotator_class_name)
                    rospy.loginfo(f"Class {annotator_class_name} found: {annotator_class}")

                # Create an instance of the annotator class
                annotator_instance = annotator_class(annotator_name)
                annotator_instance.hasServiceName.append(service_name)

                # Save the updated ontology containing the base
                # ontology_path_base = os.path.join(os.path.dirname(ONTOLOGY_PATH), updated_ontology_name + '_base.owl')
                # ontology.save(file=ontology_path_base, format="rdfxml")

                # Create a class for grouping the objects detectable
                detecting_class_name = f'EntitiesDetectectableBy{annotator_class_name}'
                try:
                    rospy.loginfo(f"Creating subclass '{detecting_class_name}' for algorithm type '{ontology.DetectableEntity}'.")
                    with ontology:
                        detecting_class = types.new_class(detecting_class_name, (ontology.DetectableEntity, ))
                except Exception as e:
                    rospy.logerr(f"Error adding class {detecting_class_name} to the ontology: {e}")


                # Check the labels created by the classes
                if detected_labels:
                    for label in detected_labels:
                        # Format labels
                        formatted_label = label.replace(' ', '_').capitalize()
                        # Add the detected property
                        if getattr(ontology, detected_property) is not None:
                            detected_property_class = getattr(ontology, detected_property)
                        else:
                            print(f"Can't find property {detected_property} in the ontology")
                            continue
                        # Ensure the labels are in the ontology
                        #if not hasattr(ontology, formatted_label) or getattr(ontology, formatted_label) is None:
                        rospy.loginfo(f"Creating subclass '{formatted_label}' for '{detected_property_class}'.")
                        with ontology:
                            if detected_property == 'ObjectType':
                                label_class = types.new_class(formatted_label, (ontology.PhysicalEntity,))
                            elif detected_property == 'ActivityType':
                                label_class = types.new_class(formatted_label, (ontology.Activity,))
                            else:
                                label_class = types.new_class(formatted_label, (detected_property_class,))                        
                            label_class = types.new_class(formatted_label, (detecting_class,))
                            label_class(f'test_{formatted_label}')
                            # Also add the class as a DetectableBy
                            # label_class = types.new_class(formatted_label, (detecting_class))

                            # types.new_class(formatted_label, (detected_property_class,))
                            # annotator_class.is_a.append(ontology.canDetect.some(label_class)) # This class axiom is removed due to not being used
                # Check if the model has an open-set labels and add physical entity as the general layer
                elif not detected_labels and detected_property == 'ObjectType':
                    types.new_class('PhysicalEntity', (detecting_class,))
                ontology.PhysicalEntity('test_PhysicalEntity')
                # Add the SWRL rules
                rospy.loginfo(f"Adding SWRL rule for {annotator_class} to the ontology...")
                try:
                    with ontology:
                        rule = Imp()
                        rule_text = """{}(?a), {}(?entity) -> canDetect(?a, ?entity)""".format(str(annotator_class).split('.')[-1], str(detecting_class).split('.')[-1])
                        rule.set_as_rule(rule_text)
                        rospy.loginfo(f"...Added SWRL rule to the ontology...")
                except Exception as e:
                    rospy.logerr(f"Failed to add SWRL rule")
                rospy.loginfo(f"Added annotator '{annotator_name}' to the ontology.")
                added_annotators.append(annotator_name)

            except AttributeError as e:
                rospy.logerr(f"Error creating or fetching class {annotator_class_name}: {e}")
            except Exception as e:
                rospy.logerr(f"Error adding annotator {annotator_name} to the ontology: {e}")


    # Save the updated ontology
    ontology_path_capability = os.path.join(os.path.dirname(ONTOLOGY_PATH), updated_ontology_name + '_capability.owl')


    try:
        with ontology:
            # Save the inferred ontology
            rospy.loginfo(f"Materializing knowledge graph...")
            sync_reasoner_pellet(infer_property_values = True, infer_data_property_values = True, debug = 0)
            # Save the inferred ontology
            rospy.loginfo(f"Saving inferred knowledge graph...")   
            ontology.save(file=ontology_path_capability, format="rdfxml")

        rospy.loginfo(f"Ontology updated and saved at {ontology_path_capability}.")
        return f"Done. Annotators added: {', '.join(added_annotators)}"
    except Exception as e:
        rospy.logerr(f"Error saving updated ontology: {e}")
        return "Error saving updated ontology."


def handle_load_models(req):
    """
    ROS service callback to load models into the ontology.

    Args:
        req: ROS service request containing updated_ontology_name.

    Returns:
        LoadModelsResponse: The response containing the status message.
    """
    rospy.loginfo("Received request to load models into ORKA.")
    updated_ontology_name = req.updated_ontology_name
    status_message = load_models(updated_ontology_name)
    return LoadModelsResponse(status=status_message)


def load_models_service():
    """
    Initialize the ROS service for loading models into ORKA.
    """
    rospy.init_node('load_models_service')
    service = rospy.Service('load_models', LoadModels, handle_load_models)
    rospy.loginfo("Load Models service is ready.")
    rospy.spin()


if __name__ == "__main__":
    try:
        load_models_service()
    except rospy.ROSInterruptException:
        pass
