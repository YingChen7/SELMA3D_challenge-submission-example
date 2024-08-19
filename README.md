# This an example for building the container image of the algorithm for SELMA3D challenge

### Step 1: Implement your solution  
* In [inference.py](inference.py), `load_image_file_as_array` function will automatically load the testing image once you submit your algorithm container. Do not change the [image reading](inference.py#L33) and [saving](inference.py#L47) parts.
  
* Modify the [processing part](inference.py#L38) in inference.py file to preprocess the input image:
  ```
  # Process the inputs: any way you'd like
    _show_torch_cuda_info()
  ```
* Put the resources required for prediction to the [resource folder](resources) such as model checkpoints, then modify the [resource part](inference.py#L41) in inference.py file to load the resources:
  ```
  with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
        print(f.read())
  ```
* Modify the [prediction part](inference.py#L44) in inference.py file, replacing it with your solution to make a prediction for the loaded image array:
  ```
  # For now, let us set make bogus predictions
  biological_brain_structure = numpy.eye(4, 2)
  ```
### Step 2: Test locally
* Call the [test_run](test_run.sh) bash script using the command:
        ```./test_run.sh```

  This will start the inference and reads from [/test/input](/test/input) and outputs to /test/output.
### Step 3: Save the container image for Grand Challenge submission
*  Call the [save](save.sh) bash script using the command:
        ```./save.sh```

   This will create a container image of the algorithm for SELMA3D challenge.
### Step 4 : Use the github repo for Grand Challenge submission (optional) 
*  Fork this repository to a new repository under your GitHub account.
*  In your repository, complete Step 1 and Step 2 to implement and test your solution.
*  Follow the instructions below for submitting to the Grand Challenge (you may need to copy and paste the link into your browser to access it):
   https://grand-challenge.org/documentation/linking-a-github-repository-to-your-algorithm/

   
