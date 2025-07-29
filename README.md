# 🐾 Wildlife-Re-ID

**Status:** 🚧 In active Development since 01.07.2025
This repository is the starting point for an open-source system for wildlife re-identification (Re-ID). It is currently under active development and will be continuously expanded and improved.

---

## 🎯 Project Goal

The goal of this project is to develop a system capable of identifying individual animals in both **images** and **video streams**. It aims to:

- Assign consistent IDs to known individuals  
- Detect and assign new IDs to previously unseen animals  
- Provide a **user-friendly interface** for managing and editing IDs (e.g., assigning names)  
- Operate reliably in **real-world conditions**, including low-resolution footage, color distortion, and limited-quality data

This tool is intended as a lightweight, open-source platform for wildlife monitoring and research. It aims to make individual animal tracking more accessible—even on limited hardware and in challenging field conditions.

---

## 🛠️ Technologies Used

> _To be added as development progresses._  
Frameworks, libraries, and tools will be listed here once selected.

---

## 🚀 Quick Start

> _Instructions coming soon._  
Setup, installation, and usage documentation will be added as the first prototype is developed.

---

## 📌 Roadmap

> _Note: This roadmap is subject to change as the project evolves._

- [x] Define the overall architecture  
- [ ] Select and evaluate baseline detection & Re-ID models  
- [ ] Develop an initial prototype for image-based re-identification
- [ ] start with only one species  
- [ ] Extend functionality to support video streams  
- [ ] Implement a GUI for editing IDs and assigning names
- [ ] expand to ID multiple species  
- [ ] Optimize performance for low-quality and real-world input

---
---

## Underlying Architecture

The Re-ID system is designed as a multi-layer pipeline that processes video frames in real-time or near real-time. Here's an overview of the planned architecture:

### 🔹 Layer 1: Motion Detection  
- Use **frame differencing** to detect motion between consecutive frames.
- Trigger analysis only when significant motion is detected to reduce unnecessary computation.

### 🔹 Layer 2: Region Proposal  
- Generate **bounding boxes** around areas with detected motion.
- Filter out regions that are too small to be relevant (e.g. noise, lighting flickers).

### 🔹 Layer 3: Animal Detection  
- **Crop** each motion region.
- Apply **pattern recognition** or a lightweight classifier to distinguish animals from false positives (e.g. swaying branches, shadows).
- If no animal is detected:
  - Discard the frame from further processing.
  - If the scene has remained stable for several frames (i.e. no significant motion or animal presence),
    **update the background reference** by using the current frame to replace the **oldest frame** in the background consensus group.
  - This helps reduce noise caused by large time differences between background frames and incoming motion, improving motion detection stability.


### 🔹 Layer 4: Re-Identification  
- If the region contains a valid animal:
  - Attempt to **match** it to previously seen individuals using a pattern-based feature extractor.
  - If a match is found:
    - Add the new image to that animal's dataset.
    - Retrain the Re-ID model incrementally, giving more weight to newer images to account for **seasonal changes** or **aging**.
  - If no match is found:
    - Register a **new animal ID** and store its pattern data for future matching.

---

This modular architecture ensures the system is lightweight, adaptive, and usable in real-world conditions where background, lighting, and image quality can vary significantly.



---

## 🤝 Contributing

Contributions are welcome! Guidelines, issue templates, and a project board will be added soon to help onboard collaborators.

---

## 📄 License

> _To be determined._  
A suitable open-source license will be chosen later in development.
