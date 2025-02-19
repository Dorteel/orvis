# How to extend ORViS
This document aims to help other developers extend ORViS with more models and service types.

## Adding new models
The necessary properties that define the model are described in the [template.yaml](/config/models/template.yaml) file.

## Adding new task types

### 1. Write a task processor class

### 2. Define new service message

### 3. Define models for the new task

### 4. Update CMakeLists.txt
Add the service file name to the `add_service_files()` function of the file.

### 5. Extend Service Manager
Change the create_service_from_config() function to add your task name as specified in your model config.
Make sure to add the appropriate task process class and the task service to the imports.

### 6. Extend Task Selector
