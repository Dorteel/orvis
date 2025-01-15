import rospy


def query_orka(query):
    """
    Queries the knowledge base
    """
    pass


def locate_object(object):
    """
    Locates the given object
    """
    # Which of the annotators is able to detect the fruit?
    capable_annotators = query_orka(object)
    
    # Try out each of the annotators
    for annotator in capable_annotators:
        # Call annotator service
        request = ObjectDetectionRequest(image=img_msg)
        response = annotator_service(request)
        



if __name__ == "__main__":

    # Food salad items
    fruit_salad_items = ['Banana', 'Apple', 'Strawberry', 'Orange', 'Pineapple']

    # 
    for fruit in fruit_salad_items:
        rospy.loginfo(f"Processing {fruit}...")
        fruit_position = locate_object(fruit)


