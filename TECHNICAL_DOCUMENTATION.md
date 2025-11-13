# VideoDoodles Technical Documentation

## Overview

VideoDoodles is a system for creating hand-drawn animations on videos with scene-aware canvases. The system consists of three main components:

1. **Preprocess**: Converts raw video data (frames, depth, cameras, flows) into formats optimized for the application
2. **Frontend**: Web-based UI built with Svelte and Three.js for interactive canvas creation and animation
3. **Backend**: Python WebSocket server that performs 3D trajectory tracking and optimization

---

## 1. Preprocess Component

### Purpose
The preprocessing pipeline converts raw video data into structured formats that enable efficient 3D tracking and rendering in the application. It processes video frames, depth maps, camera parameters, optical flows, and generates deep image features.

### Architecture

#### Main Entry Point: `prepare_all.py`
Orchestrates the entire preprocessing pipeline through 6 sequential steps:

1. **Frame Preparation** (`frames.py`)
2. **Camera Preparation** (`cameras.py`)
3. **Depth Map Preparation** (`depth_maps.py`)
4. **Deep Image Feature Extraction** (`videowalk_features.py`)
5. **3D Tracking Maps Generation** (`tracking_maps.py`)
6. **Tracking Masks Generation** (`tracking_masks.py`)

#### Input Data Format

The preprocessing expects raw data in the following structure:
```
raw-data/
    video_name/
        frames.npz              # Video frames (RGB images)
        flows.npz               # Optical flow between consecutive frames
        flows_con.npz           # Flow consistency masks
        resized_disps.npz       # Disparity maps (inverse depth)
        refined_cameras.txt     # Camera parameters per frame
```

**Key Input Files:**
- `frames.npz`: Contains arrays `frame_XXXXX` for each frame (shape: `(height, width, 3)`)
- `flows.npz`: Contains `flow_XXXXX_to_YYYYY` arrays (2D optical flow vectors)
- `flows_con.npz`: Contains `consistency_XXXXX_YYYYY` binary masks indicating flow reliability
- `resized_disps.npz`: Contains `disp_XXXXX` disparity maps (shape: `(height, width, 1)`)
- `refined_cameras.txt`: Text file with camera parameters (translation, rotation, scale, shift)

#### Output Data Formats

The preprocessing generates two sets of output data:

##### A. Backend Data (`app/backend/data/video_name/`)

**1. 3D Position and Flow Maps:**
- `maps_dim.npy`: Array dimensions `(nb_frames, maps_res_x, maps_res_y)`
- `pos.memmap`: Memory-mapped array of 3D world-space points
  - Shape: `(nb_frames, maps_res_x * maps_res_y, 3)`
  - Dtype: `float64`
  - Each pixel in each frame has a corresponding 3D position
- `flow.memmap`: Memory-mapped array of 3D scene flow vectors
  - Shape: `(nb_frames, maps_res_x * maps_res_y, 3)`
  - Dtype: `float64`
  - Represents 3D motion between consecutive frames

**2. Deep Image Features:**
- `features_dim.npy`: Array dimensions `(nb_frames, feats_res_x, feats_res_y, latent_dim)`
- `features.memmap`: Memory-mapped array of deep image features
  - Shape: `(nb_frames, feats_res_x * feats_res_y, latent_dim)`
  - Dtype: `float32`
  - Extracted using VideoWalk (CRW model) for appearance-based tracking

**3. Camera Parameters:**
- `cameras.npz`: Contains:
  - `Ks`: Camera intrinsics matrices (shape: `(nb_frames, 3, 3)`)
  - `Rs`: Rotation matrices (shape: `(nb_frames, 3, 3)`)
  - `ts`: Translation vectors (shape: `(nb_frames, 3)`)
  - `res`: Video resolution `(width, height)`
  - `down_scale_factor`: Scale factor between 3D scene and UI scale
  - `near`: Camera near plane
  - `far`: Camera far plane

**4. Tracking Masks:**
- `masks.npz`: Contains boolean array `masks`
  - Shape: `(nb_frames, maps_res_x * maps_res_y)`
  - Indicates pixels with untrustworthy 3D flow (based on flow consistency)

##### B. Frontend Data (`app/frontend/public/data/video_name/`)

**1. Video File:**
- `vid.mp4`: Encoded video file optimized for frame-by-frame scrubbing

**2. Depth Maps:**
- `depth_16/XXXX.png`: 16-bit depth maps encoded as 8-bit PNG images
  - Depth values normalized to [0, 1] range
  - Encoded using blue and green channels (16 bits split across 2 channels)
  - Used for depth compositing in Three.js rendering

**3. Camera Data:**
- `camera.json`: JSON array with one camera object per frame:
  ```json
  {
    "rotation": [x, y, z, w],  // Quaternion
    "translation": [x, y, z],  // 3D position
    "cameraProjectionTransform": [...],  // 4x4 projection matrix (row-major)
    "depthRange": float,       // Depth normalization range
    "depthOffset": float       // Depth normalization offset
  }
  ```

### Key Processing Steps

#### 1. Frame Preparation (`frames.py`)
- Extracts RGB frames from `frames.npz`
- Encodes frames into MP4 video format optimized for frame-by-frame access
- Output: `vid.mp4` in frontend data folder

#### 2. Camera Preparation (`cameras.py`)
- Parses camera parameters from `refined_cameras.txt`
- Converts from COLMAP format (right-handed, z-forward, -y-up) to OpenGL format
- Generates projection matrices for Three.js rendering
- Exports both backend (NumPy) and frontend (JSON) formats
- **Outputs:**
  - Backend: `cameras.npz` with intrinsics, rotations, translations
  - Frontend: `camera.json` with OpenGL-compatible camera data

#### 3. Depth Map Preparation (`depth_maps.py`)
- Converts disparity maps to depth maps: `depth = 1.0 / disparity`
- Normalizes depth values using camera scale/shift parameters
- Encodes 16-bit depth values into 8-bit PNG images (using bit packing)
- **Output:** `depth_16/XXXX.png` files for each frame

#### 4. Deep Image Feature Extraction (`videowalk_features.py`)
- Uses VideoWalk (CRW) model to extract pixel-level deep features
- Processes frames through ResNet-based encoder
- Reduces resolution to 480px height for efficiency
- Stores features as memory-mapped arrays for efficient access
- **Output:** `features.memmap` and `features_dim.npy`

#### 5. 3D Tracking Maps (`tracking_maps.py`)
- Unprojects depth maps to 3D point clouds using camera parameters
- Computes 3D scene flow from optical flow and depth
- Creates memory-mapped arrays for efficient random access
- **Output:** `pos.memmap`, `flow.memmap`, `maps_dim.npy`

#### 6. Tracking Masks (`tracking_masks.py`)
- Generates binary masks indicating unreliable flow regions
- Based on flow consistency checks (forward/backward flow agreement)
- **Output:** `masks.npz`

### Data Flow: Preprocess → Application

**Preprocessing data is used by the application as follows:**

1. **Backend uses:**
   - `pos.memmap` + `flow.memmap`: For 3D trajectory tracking (finding motion paths through video volume)
   - `features.memmap`: For appearance-based matching in tracking
   - `cameras.npz`: For coordinate transformations and 3D point unprojection
   - `masks.npz`: To filter out unreliable tracking regions

2. **Frontend uses:**
   - `vid.mp4`: For video playback and frame display
   - `depth_16/*.png`: For depth compositing (occlusion handling) in Three.js
   - `camera.json`: For setting up Three.js camera parameters per frame

---

## 2. Frontend Component (`app/frontend`)

### Purpose
The frontend provides an interactive web-based interface for creating and animating hand-drawn canvases on videos. Users can draw on frames, set keyframes, and the system automatically tracks canvas positions and orientations through 3D space.

### Technology Stack
- **Framework**: Svelte (reactive UI framework)
- **3D Rendering**: Three.js (WebGL-based 3D graphics)
- **Build Tool**: Webpack
- **Communication**: WebSocket (for backend communication)

### Architecture

#### Core Components

**1. Main Application (`App.svelte`, `Main.svelte`)**
- Root component managing application state
- Handles routing and top-level UI layout

**2. Viewport System (`Viewport.svelte`, `Frame.js`)**
- Three.js scene rendering
- Composites video frames with 3D-rendered canvases
- Handles depth-based occlusion
- Manages camera setup per frame using `camera.json` data

**3. Canvas System (`Canvas.js`, `AnimatedCanvas.js`)**
- Represents a drawable canvas in 3D space
- Manages trajectory (position + orientation per frame)
- Handles keyframe creation and editing
- Stores drawing strokes and canvas properties

**4. Drawing System (`DrawingCanvas.svelte`, `SceneViewFromCanvas.svelte`)**
- Provides drawing interface with scene-aware view
- Remaps 3D scene view "from the canvas" perspective
- Enables sketching in context

**5. Timeline System (`Timeline.svelte`, `KeyframeTimeline.svelte`)**
- Displays and navigates video frames
- Shows keyframes for each canvas
- Allows frame-by-frame scrubbing

**6. Animation Controller (`AnimationController.js`)**
- Manages all animated canvases
- Coordinates trajectory updates from backend
- Handles animation playback

**7. WebSocket Client (`websocket.js`)**
- Manages WebSocket connection to backend
- Sends trajectory inference requests
- Receives and processes trajectory results

### Data Structures

#### Canvas Keyframe Format
```javascript
{
  time: number,              // Frame index
  props: {
    x: number,               // Normalized x position [0, 1]
    y: number,               // Normalized y position [0, 1]
    depth?: number,          // Optional depth value
    rot?: {                  // Optional rotation matrix
      elements: number[]     // 4x4 matrix (row-major)
    }
  }
}
```

#### Trajectory Point Format
```javascript
{
  position: Vector3,         // 3D position
  rotation: Matrix4,         // 3D rotation (4x4 matrix)
  positionSolved: boolean,   // Whether position was computed
  rotationSolved: boolean    // Whether rotation was computed
}
```

#### WebSocket Message Format (Frontend → Backend)

**Trajectory Inference Request:**
```javascript
{
  action: "INFER_TRAJECTORY",
  canvasID: number,
  clip: string,              // Video name
  type: "static" | "dynamic",
  keyframes: [               // Array of keyframes
    {
      time: number,
      props: {
        x: number,
        y: number,
        depth?: number,
        rot?: { elements: number[] }
      }
    }
  ],
  positionSegments: [        // Frame ranges for position tracking
    {
      start: number,
      end: number,
      dirty: boolean,        // Whether segment needs update
      mode: number           // 0 = interpolation, 1 = tracking
    }
  ],
  orientationSegments: [     // Frame ranges for orientation tracking
    {
      start: number,
      end: number,
      dirty: boolean,
      mode: number
    }
  ]
}
```

**Export Keyframes Request:**
```javascript
{
  action: "EXPORT_KEYFRAMES",
  canvasID: number,
  clip: string,
  type: string,
  keyframes: [...],
  positionSegments: [...],
  orientationSegments: [...]
}
```

#### WebSocket Message Format (Backend → Frontend)

**Position Trajectory Result:**
```javascript
{
  status: "ESTIMATION_POSITION_SUCCESS",
  message: string,
  canvasID: number,
  frameIndices: number[],    // Frame indices updated
  positions: [               // 3D positions [x, y, z]
    [number, number, number],
    ...
  ]
}
```

**Orientation Trajectory Result:**
```javascript
{
  status: "ESTIMATION_ORIENTATION_SUCCESS",
  message: string,
  canvasID: number,
  frameIndices: number[],
  orientations: [            // 3x3 rotation matrices (flattened)
    [number, number, number, ...],  // 9 numbers per matrix
    ...
  ]
}
```

**Error/Status Messages:**
```javascript
{
  status: "ERROR" | "ESTIMATION_FAILURE" | "ESTIMATION_POSITION_UNCHANGED" | "ESTIMATION_ORIENTATION_UNCHANGED",
  message: string,
  canvasID?: number,
  frameIndices?: number[]
}
```

### Rendering Pipeline

1. **Frame Loading**: Loads video frame and corresponding depth map
2. **Camera Setup**: Configures Three.js camera using `camera.json` data
3. **Depth Compositing**: Uses depth maps to handle occlusion between video and 3D objects
4. **Canvas Rendering**: Renders canvas strokes as 3D objects at tracked positions
5. **SVG Overlay**: Draws 2D gizmos (handles, controls) on top of 3D render

### Data Flow: Frontend ↔ Backend

**Frontend → Backend:**
1. User creates keyframes by drawing on frames
2. User triggers trajectory inference
3. Frontend sends keyframe data + segments to backend via WebSocket
4. Backend processes and returns trajectory results

**Backend → Frontend:**
1. Backend sends position trajectory (array of 3D positions)
2. Backend sends orientation trajectory (array of rotation matrices)
3. Frontend updates canvas trajectory and re-renders
4. Frontend marks trajectory points as "solved"

---

## 3. Backend Component (`app/backend`)

### Purpose
The backend performs 3D trajectory tracking and optimization. Given user-specified keyframes (2D positions + optional depth/rotation), it computes smooth 3D trajectories that follow scene motion through the video.

### Technology Stack
- **Language**: Python 3.10
- **Web Framework**: WebSockets (asyncio-based)
- **Scientific Computing**: NumPy, SciPy
- **Optimization**: Sparse linear solvers, graph algorithms

### Architecture

#### Main Server (`app.py`)
- WebSocket server listening on port 8001
- Handles incoming requests from frontend
- Manages per-canvas state
- Coordinates trajectory solving pipeline

#### Core Modules

**1. Trajectory Solving (`solve_trajectory.py`)**
- Main entry point for trajectory computation
- Orchestrates position and orientation solving
- Handles static vs. dynamic movement types
- Manages incremental updates (only recomputes dirty segments)

**2. Position Tracking (`tracking_position.py`)**
- **Motion Path Finding** (`find_motion_path`):
  - Constructs directed graph connecting pixels across frames
  - Each node = (frame, pixel), edges = 3D scene flow
  - Uses shortest path algorithm to find trajectory through video volume
  - Considers 3D proximity, feature similarity, and keyframe constraints
- **Trajectory Optimization** (`optimize_trajectory`):
  - Refines initial motion path using sparse linear solver
  - Enforces smoothness constraints
  - Respects keyframe constraints
  - Outputs high-resolution, stable trajectory

**3. Orientation Tracking (`tracking_orientation.py`)**
- Optimizes canvas orientations given position trajectory
- Uses velocity vectors from position tracking to guide orientation
- Enforces smooth rotation transitions
- Respects user-specified orientation keyframes

**4. Scene Data Access (`read_scene_data.py`)**
- Provides interface to preprocessed data
- Memory-maps large arrays for efficient access
- Handles coordinate transformations
- Indexes into 3D position/flow maps from 2D pixel coordinates

**5. Data Conversion (`convert.py`)**
- Converts between frontend (JSON) and backend (NumPy) formats
- Handles coordinate system transformations (OpenGL ↔ OpenCV)
- Unprojects 2D+depth to 3D positions
- Parses trajectory data from WebSocket messages

**6. State Management (`state_management.py`)**
- Tracks per-canvas state across requests
- Stores previous trajectory solutions for incremental updates
- Manages velocity constraints and matching weights

### Algorithm Overview

#### Position Tracking Algorithm

1. **Motion Graph Construction:**
   - For each frame, create nodes for all pixels (at tracking resolution)
   - Connect nodes between consecutive frames using 3D scene flow
   - Edge weights based on:
     - 3D distance (proximity_weight)
     - Feature similarity (feature_similarity_weight)
     - Distance to keyframe targets (targets_feature_similarity_weight)

2. **Path Finding:**
   - Find shortest path through graph from first to last keyframe
   - Prune nodes/edges to reduce computation
   - Output: Initial 3D trajectory (one point per frame)

3. **Trajectory Optimization:**
   - Set up sparse linear system:
     - Data term: Match keyframe constraints
     - Smoothness term: Minimize acceleration
     - Soft velocity constraint: Encourage following scene flow
   - Solve using sparse linear solver
   - Output: Optimized, high-resolution trajectory

#### Orientation Tracking Algorithm

1. **Velocity Analysis:**
   - Extract velocity vectors from position trajectory
   - Compute matching weights (based on velocity magnitude)
   - Identify motion segments

2. **Orientation Optimization:**
   - For each motion segment:
     - Align canvas forward direction with velocity
     - Enforce smooth rotation transitions
     - Respect user-specified orientation keyframes
   - Use SLERP for interpolation between keyframes

### Data Flow: Backend Processing

**Input (from Frontend):**
- Keyframes: 2D positions (x, y) + optional depth + optional rotation
- Movement type: "static" or "dynamic"
- Frame segments: Ranges that need tracking

**Processing:**
1. Parse and convert keyframe data
2. Load preprocessed scene data (positions, flows, features, cameras)
3. For static movement: Use first keyframe position for all frames
4. For dynamic movement:
   - Find motion path through video volume
   - Optimize trajectory
   - Compute orientations based on velocity
5. Convert results back to frontend format

**Output (to Frontend):**
- Position trajectory: Array of 3D positions (one per frame)
- Orientation trajectory: Array of 3x3 rotation matrices (one per frame)
- Status messages indicating success/failure

### State Management

The backend maintains state per canvas to enable incremental updates:

```python
state_per_canvas[unique_ID(video_name, canvas_id)] = {
    "positions": np.ndarray,      # Previous position trajectory
    "velocities": np.ndarray,     # Velocity vectors
    "orientations": np.ndarray,   # Previous orientations
    "orientation_matching_weights": np.ndarray,  # For orientation tracking
    "indices": np.ndarray         # Frame indices
}
```

This allows:
- Only recomputing "dirty" segments when keyframes change
- Reusing previous solutions as initialization
- Maintaining smooth transitions between segments

### Data Access Patterns

**Reading Preprocessed Data:**
- Uses memory-mapped arrays (`np.memmap`) for efficient access
- Loads data on-demand (not all at once)
- Indexes into arrays using flattened pixel coordinates

**Example: Getting 3D position for a 2D pixel:**
```python
# Convert 2D pixel (normalized [0,1]) to data resolution
flat_idx = index_into_data(pixel, pixel_res, data_res)
# Read from memory-mapped array
pos_3d = pos_3d_archive[frame_idx, flat_idx]
```

---

## 4. Data Flow Summary

### Preprocessing → Application

**Backend receives:**
- `pos.memmap`: 3D point clouds per frame (for motion path finding)
- `flow.memmap`: 3D scene flow vectors (for graph edge weights)
- `features.memmap`: Deep image features (for appearance matching)
- `cameras.npz`: Camera parameters (for coordinate transformations)
- `masks.npz`: Flow reliability masks (for filtering)

**Frontend receives:**
- `vid.mp4`: Video file (for frame display)
- `depth_16/*.png`: Depth maps (for occlusion handling)
- `camera.json`: Camera parameters (for Three.js rendering)

### Frontend ↔ Backend Communication

**Frontend → Backend (WebSocket):**
- **Keyframes**: User-drawn positions/rotations at specific frames
- **Segments**: Frame ranges that need tracking updates
- **Movement type**: Static or dynamic
- **Purpose**: Request 3D trajectory computation

**Backend → Frontend (WebSocket):**
- **Position trajectory**: 3D positions for each frame
- **Orientation trajectory**: 3D rotations for each frame
- **Status messages**: Success/failure/unchanged notifications
- **Purpose**: Update canvas animation with computed trajectories

### Data Transformations

1. **2D → 3D (Frontend → Backend):**
   - User draws at 2D pixel (x, y) with optional depth
   - Backend unprojects to 3D using camera parameters
   - Uses depth map if depth not provided

2. **3D Tracking (Backend):**
   - Finds path through 3D video volume
   - Optimizes trajectory in 3D space
   - Computes orientations aligned with motion

3. **3D → 2D Rendering (Backend → Frontend):**
   - 3D trajectory positions/orientations
   - Frontend projects to screen using camera parameters
   - Renders canvas strokes at tracked positions

---

## 5. Key Design Decisions

### Memory-Mapped Arrays
- Large 3D position/flow/feature arrays use memory-mapped files
- Enables efficient random access without loading entire arrays
- Critical for handling large videos

### Incremental Updates
- Backend tracks per-canvas state
- Only recomputes "dirty" segments when keyframes change
- Enables interactive editing without full recomputation

### Separate Frontend/Backend Data
- Frontend and backend have separate data folders
- Avoids shared storage dependencies
- Allows independent deployment

### Coordinate System Handling
- Preprocessing converts COLMAP format to OpenGL format
- Backend uses OpenCV-style coordinates internally
- Frontend uses Three.js (OpenGL) coordinates
- Conversion functions handle transformations

### WebSocket Communication
- Real-time bidirectional communication
- Enables incremental trajectory updates
- Supports long-running computations with progress updates

---

## 6. File Structure Reference

```
VideoDoodles/
├── preprocess/              # Preprocessing scripts
│   ├── prepare_all.py       # Main entry point
│   ├── cameras.py           # Camera data processing
│   ├── depth_maps.py        # Depth map encoding
│   ├── frames.py            # Video frame encoding
│   ├── tracking_maps.py     # 3D position/flow maps
│   ├── tracking_masks.py    # Flow reliability masks
│   └── videowalk_features.py # Deep feature extraction
│
├── app/
│   ├── backend/             # Python WebSocket server
│   │   ├── app.py           # Main server
│   │   ├── scripts/
│   │   │   ├── solve_trajectory.py      # Trajectory solving
│   │   │   ├── tracking_position.py     # Position tracking
│   │   │   ├── tracking_orientation.py  # Orientation tracking
│   │   │   ├── read_scene_data.py       # Data access
│   │   │   ├── convert.py               # Format conversion
│   │   │   └── state_management.py      # State tracking
│   │   └── data/            # Backend preprocessed data
│   │
│   └── frontend/            # Svelte web application
│       ├── src/
│       │   ├── App.svelte   # Root component
│       │   ├── Main.svelte  # Main UI
│       │   ├── Viewport.svelte          # 3D rendering
│       │   ├── AnimatedCanvas.js        # Canvas logic
│       │   ├── AnimationController.js   # Animation management
│       │   ├── websocket.js             # Backend communication
│       │   └── components/              # UI components
│       └── public/data/     # Frontend preprocessed data
│
└── raw-data/                # Input raw video data
```

---

## 7. Usage Flow

1. **Preprocessing:**
   ```bash
   python3 preprocess/prepare_all.py --vid video_name -E
   ```
   Generates backend and frontend data from raw video

2. **Backend Server:**
   ```bash
   cd app/backend
   python3 app.py
   ```
   Starts WebSocket server on port 8001

3. **Frontend Application:**
   ```bash
   cd app/frontend
   npm run dev
   ```
   Starts web server on port 8080

4. **User Workflow:**
   - Load video in frontend
   - Create canvas
   - Draw on frames to set keyframes
   - Trigger trajectory inference
   - Backend computes 3D trajectory
   - Frontend displays animated canvas
   - Export result

---

This documentation provides a comprehensive overview of the VideoDoodles codebase architecture, data flow, and component interactions.

