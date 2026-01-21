# niryo-chess-bot
## 1. Overview
This project enables the Niryo robot Ned2 to play a game of chess against a human player. This repo explains the process of making the board and pieces needed for the robot to play. Then the environment setup will be explained as well as how the code runs.
## 2. Required Material
For this project, you will need :
  * Niryo Ned2 Robot
  * Vision Set
  * Eletro-magnet for the robot
  * 3D Printer
  * Black and white filaments
  * Regular Printer (sticker printer would be better)
  * 10mm x 1mm magnets (at least 96 [64 tiles + 32 pieces])
  * 13mm x 1mm steel washers (at least 32 for each pieces)

## 3. Material recreation
### 3.1 The board
The board is made of 64 separated tiles that you can print individually. The .stl are located in ```/assets/3D/board```. You will need to print :
  * x30 black main_tile.stl
  * x30 white main_tile.stl
  * x2 black corner_tile.stl
  * x2 white corner_tile.stl
  * x112 white or black tile_clipper.stl

<img width="227" alt="image" src="https://github.com/user-attachments/assets/41d1c6e1-e33b-4a82-bda6-e57a21a21c1b" />
    
**Note : corner_tile.stl is a bit different thant the other tiles. It features a thin identation for the NiryoMarker emplacement**

### 3.2 Pieces
You can find .stl files in ```/assests/3D/pieces```. If you're using FDM printer, set the infill to 80% or more to get a better feeling and wheight while manipulating the pieces. I used SLA printer for the pieces (resin type printer).

<img width="259" alt="image" src="https://github.com/user-attachments/assets/737ccf64-fa41-4180-88f9-dc05c0db5ef4" />


### 3.3 Niryo Markers & Stickers
For the robot workspace visualisation, print Niryo_Markers in ```/assets/2D/``` and stick them in the 4 corners of the board. Note that the one different marker form the others should be on a white tile and on top left corner when looking towards the robot.

For the pieces stickers, print them via ```/assets/2D/sticker_chess``` and place them on top of each pieces on the steel washers.

### 3.4 Assembly
The player can move a piece inaccurately on a tile and the robot can have issues lifting it up. So we place magnets in the board and under the pieces to solve this issue. The holes are planned in the .stl files for the 10x1mm neodymium magnets.

```mermaid
flowchart TB
  subgraph PIECE
    style P fill:#f9f,stroke:#333,stroke-width:2px
    P([PIECE])
    M1([N]):::magnet_piece
    P -->|magnet| M1
  end

  subgraph BOARD
    direction LR
    M2([S]):::magnet_board
    M3([S]):::magnet_board
    M4([S]):::magnet_board
    M2 & M3 & M4
  end

  M1 -- Attracts --> M2

  classDef magnet_piece fill:#faa,stroke:#900,stroke-width:1px;
  classDef magnet_board fill:#aaf,stroke:#009,stroke-width:1px;

```


N -> North magnet pole on the piece

S -> South magnet pole on the board tiles


## 4. Environment setup
### 4.1 Requirements 
To run ```pipeline.py```, you need to execute the script in a conda environment https://anaconda.org/anaconda/conda 

Then install the environment setup
```
conda env create-f environment.yml
``` 

### 4.2 Config
There might be changes to apply to constants in pipeline.py :
  * If you changed the pieces, change the ```PIECE_HEIGHT``` values.
  * If you changed the board, change the ```CELL_SIZE``` to yours.
  * Change ```ROBOT_IP``` to yours if needed (default 10.10.10.10)

### 4.3 Import audio files
In futuer realses, the robot will pronounce the move it plays. Sounds has to be uploaded into robot's memory card.
Use 
```
python import_sounds.py
```
to upload audio files towards Ned2.

### 4.4 Load AI Model
```pipieline.py``` uses torch to load the next move prediction AI model.
Download the ```model_data``` file. Unzip it and place the model_data folder in ```ChessUtils/```

### 4.5 Calibrate the workspace
Download NiryoStudio : https://niryo.com/niryostudio/

Connect to the robot with the application and got to ``` Library > Workspaces > Add+ ``` and calibrate the new workspace. **Make sur to name the workspace ```"ChessBoard"```**


### 4.6 (Optionnal) Train your own model
If you have different pieces than the .stl ones provided, you might want to train your own pieces detection model.
I used YOLO v5 model. Annotated data with LabelStudio and trained in Google Colab instace.
Save the model weights (best.pt or last.pt) and place it in ```src/```








