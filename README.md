# Hand-Tool Kitting Dataset
<br>
Nanyang Technological University
<br>

  
## Environment
```
conda env create -f environment.yml
conda activate telerobot
```

## Dataset
1. Download hand-tool models by running [get_toolbox_cad.sh](./get_toolbox_cad.sh), then the cad folder _**toolbox_cad**_ will be saved into the [assets](./assets/) folder
      ```sh
      ./get_toolbox_cad.sh
      ```

2. Modify [tool_kit.yaml](./conf/data_gen/tool_kit.yaml) to set the train/test dataset, and the file location when necessary

3. Running [prepare_tool_kit.py](./prepare_tool_kit.py) to start the data generation, and the file will be saved in the _**toolbox_cad**_ folder.
      ```
      python prepare_tool_kit.py
      ```

