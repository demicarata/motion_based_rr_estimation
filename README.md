# Motion-Based Respiratory Rate Extraction Algorithms

This project includes a main program that can run one of three different algorithms on a video of an infant subject to extract the respiratory rate. The extracted values are compared to a ground-truth file, after which statistics about this comparison are added to a .csv file, for further analysis.
This work is part of the Research Project at TU Delft.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Algorithm Options](#algorithm-options)

## Prerequisites
- Python 3.12

## Installation
1. Clone this repository
2. Install the required packages by using this command:
   ```bash
   pip install -r requirements.txt
   
## File Structure
The project follows a modular structure with a main file to run the program, three algorithms implemented in separate files, as well as utility functions.


```
    ResearchProject/
    │
    ├── main.py                  # Entry point of the project
    │
    ├── algorithms/              # Contains the implementations of the three algorithms
    │   ├── pic.py               # Implementation of the Pixel Intensity Changes algorithm.
    │   ├── of.py                # Implementation of the Optical Flow algorithm
    │   └── evm.py               # Implementation of the Eulerian Video Magnification algorithm
    │
    ├── helper/                  # Folder containing utility/helper functions
    │   ├── calculate_RR.py      # Provides methods of extracting the respiratory rate in BPM from a signal
    │   ├── filters.py           # Contains implementations for the various signal filtering methods used in this project
    │   ├── ground_truth.py      # Contains the logic behind processing the ground_truth respiratory impulse into RR
    │   └── visualisation.py     # Provides functions for plotting different data
    │
    ├── statistics.py            # Compiles the results of the algorithm runs
    ├── algorithm_results.csv    # Stores the statistics extracted after each run of the algorithm
    ├── requirements.txt         # List of dependencies
    ├──README.md                 # This file
    │
    └── AIR_converted/           # Dataset used for testing
        ├── S01/                 # Data for the first subject
        │   ├── hdf5/            # Holds ground truth data
        │   │   ├── 001.hdf5     # Ground truth file
        │   │   └── ...
        │   └── videos/          # Holds the videos used for testing
        │       ├── 001_720p.mp4 # Video file
        │       └── ...
        ├── S02/
        └── ...
```

## Usage
1. Change the path to the video and the ground-truth file in `main.py`. Ensure that the ground_truth file corresponds to the video.
2. Run the program using 
    ```bash 
   python main.py --algorithm [algorithm of your choice]
3. The Region Of Interest Selection(ROI) screen will open
4. Select the desired ROI by clicking and dragging on the screen. Press Enter to proceed. Click outside the selected ROI to cancel the selection.
5. The algorithm will run on the selected ROI until the end. After 10 seconds, the Respiratory Rate will update on the screen every 1 second.
6. After the video is finished, plots comparing the extracted respiratory rate to the ground-truth, as well as the computational complexity, will appear on the screen.
7. The `algorithms.csv` file will be updated with the name of the algorithm used, the video path, the fps, the processing time per frame, the calculated Pearson's coefficient and the Root Mean Squared Error.
8. The results after running the algorithms multiple times can be analysed by running the `statistics.py` file.
    ```
   python statistics.py
   
## Algorithm Options
The program provides three algorithms to choose from:

### Pixel Intensity Changes
Calculates the motion signal by analysing the difference in individual pixel intensities across the video.
- Run:
   ```bash
   python main.py --algorithm 1

### Optical Flow
Analyses the motion of tracking points between consecutive frames.
- Run:
   ```bash
   python main.py --algorithm 2

### Eulerian Video Magnification
Enhances the respiratory movements in a certain frequency, and then calculates the respiratory rate based on them. 
- Run:
   ```bash
   python main.py --algorithm 3

## License

This project is licensed under the MIT License. However, all third-party libraries used are governed by their own licenses.

You can view the full license here: [MIT License](https://opensource.org/license/mit) or in the [LICENSE](LICENSE) file.
