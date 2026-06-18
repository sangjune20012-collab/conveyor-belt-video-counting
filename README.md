# \# Conveyor Belt Video Counting

# 

# Segmentation-based conveyor belt object tracking and counting using SAM3/SAM3.1 video masks.

# 

# \## Overview

# 

# This repository provides a conveyor belt video counting pipeline based on video segmentation tracking.

# 

# The project started from vehicle tracking experiments, where YOLO box tracking and SAM3 mask tracking were compared on common classes such as cars and trucks. The pipeline was then extended to conveyor belt videos, where small industrial objects such as bolts are tracked and counted using SAM3/SAM3.1 segmentation masks.

# 

# The final goal is conveyor belt object counting, not general traffic counting.

# 

# \## Pipeline

# 

# ```text

# Input conveyor belt video

# &#x20;       ↓

# SAM3 / SAM3.1 video segmentation tracking

# &#x20;       ↓

# Object mask extraction

# &#x20;       ↓

# Mask centroid calculation

# &#x20;       ↓

# Polygon-zone entry detection

# &#x20;       ↓

# Object counting

# &#x20;       ↓

# Visualization video

# ```

# 

# \## Results

# 

# \### SAM3 Bolt Tracking

# 

# !\[SAM3 Bolt Tracking](docs/images/conveyor\_sam3\_bolt\_tracking.jpg)

# 

# \### SAM3.1 Bolt Tracking

# 

# !\[SAM3.1 Bolt Tracking](docs/images/conveyor\_sam31\_bolt\_tracking.jpg)

# 

# \### SAM3 Polygon-Zone Counting

# 

# !\[SAM3 Polygon Counting](docs/images/conveyor\_sam3\_polygon\_counting.jpg)

# 

# \## Repository Structure

# 

# ```text

# conveyor-belt-video-counting/

# ├─ scripts/

# │  ├─ sam3\_segment\_video.py

# │  ├─ sam31\_segment\_video.py

# │  └─ polygon\_zone\_counting.py

# │

# ├─ prototypes/

# │  └─ vehicle\_tracking\_prototype.py

# │

# ├─ docs/

# │  └─ images/

# │

# ├─ examples/

# │  └─ conveyor\_belt\_bolt/

# │

# ├─ README.md

# ├─ requirements.txt

# └─ .gitignore

# ```

# 

# \## Main Scripts

# 

# | Script                                     | Description                                                       |

# | ------------------------------------------ | ----------------------------------------------------------------- |

# | `scripts/sam3\_segment\_video.py`            | Runs SAM3 video segmentation tracking and saves mask results.     |

# | `scripts/sam31\_segment\_video.py`           | Runs SAM3.1 video segmentation tracking and saves mask results.   |

# | `scripts/polygon\_zone\_counting.py`         | Counts objects using saved segmentation masks and a polygon zone. |

# | `prototypes/vehicle\_tracking\_prototype.py` | Early YOLO/SAM3 vehicle tracking comparison prototype.            |

# 

# \## Setup

# 

# This repository does not include the official SAM3 source code, checkpoints, input videos, or generated output videos.

# 

# Install dependencies:

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# Install SAM3 separately:

# 

# ```bash

# git clone <OFFICIAL\_SAM3\_REPOSITORY\_URL>

# cd sam3

# pip install -e .

# ```

# 

# Download the required SAM3/SAM3.1 checkpoints separately and pass their paths through command-line arguments.

# 

# \## Usage

# 

# \### 1. SAM3 video segmentation tracking

# 

# ```bash

# python scripts/sam3\_segment\_video.py \\

# &#x20; --video "/path/to/conveyor\_video.avi" \\

# &#x20; --output-dir "/path/to/output" \\

# &#x20; --sam3-ckpt "/path/to/sam3.pt" \\

# &#x20; --sam3-gpu 0 \\

# &#x20; --text "bolt" \\

# &#x20; --scale 1.0

# ```

# 

# \### 2. SAM3.1 video segmentation tracking

# 

# ```bash

# python scripts/sam31\_segment\_video.py \\

# &#x20; --video "/path/to/conveyor\_video.avi" \\

# &#x20; --output-dir "/path/to/output" \\

# &#x20; --sam3p1-ckpt "/path/to/sam3.1\_multiplex.pt" \\

# &#x20; --sam3p1-gpu 0 \\

# &#x20; --text "bolt" \\

# &#x20; --scale 1.0

# ```

# 

# \### 3. Polygon-zone counting

# 

# ```bash

# python scripts/polygon\_zone\_counting.py \\

# &#x20; --frames-dir "/path/to/output/frames" \\

# &#x20; --sam-pkl "/path/to/output/sam3/sam3\_results.pkl" \\

# &#x20; --output "/path/to/output/bolt\_polygon\_count.mp4" \\

# &#x20; --fps 25 \\

# &#x20; --polygon "750,550" "1650,550" "1650,1150" "750,1150" \\

# &#x20; --title "SAM3 Bolt Polygon Counting" \\

# &#x20; --count-mode enter

# ```

# 

# \## Counting Logic

# 

# The current counting logic uses the centroid of each segmentation mask.

# 

# ```text

# Object mask

# &#x20;       ↓

# Mask centroid

# &#x20;       ↓

# Polygon inside/outside check

# &#x20;       ↓

# Outside-to-inside transition

# &#x20;       ↓

# Count once per object ID

# ```

# 

# Supported count modes:

# 

# | Mode          | Description                                                     |

# | ------------- | --------------------------------------------------------------- |

# | `enter`       | Count when the object centroid enters the polygon from outside. |

# | `inside\_once` | Count once when the object centroid is inside the polygon.      |

# 

# \## Notes

# 

# \* Input videos are not included.

# \* SAM3/SAM3.1 checkpoints are not included.

# \* Generated result videos are not included.

# \* The official SAM3 repository must be installed separately.

# \* SAM3/SAM3.1 video outputs do not provide YOLO-style class confidence scores in the current API path.

# 

# \## Limitations

# 

# \* Counting depends on mask centroid stability.

# \* ID switches can cause duplicate counts.

# \* Short segmentation failures can affect counting.

# \* More stable counting can be added using minimum inside-frame thresholds and mask-polygon overlap ratios.

# 

