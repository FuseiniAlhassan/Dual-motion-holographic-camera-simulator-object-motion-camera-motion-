This project presents a dynamic simulation of digital holography that models both object motion and camera oscillation in a Fresnel diffraction framework. A grayscale image is treated as a complex object field and subjected to controlled translation, rotation, and scaling over time, simulating physical movement. Simultaneously, the virtual camera plane oscillates along the optical axis, introducing depth variation in the hologram formation process.
At each time step, the object field is numerically propagated to the sensor plane using a Fresnel transfer function. A reference wave either on-axis or off-axis interferes with the propagated object wave to produce a time-varying hologram. Reconstruction is performed by back-propagating the recorded intensity using the conjugate transfer function, yielding a dynamic view of the recovered object field.
The simulation visualizes three panels per frame: the moving object’s intensity, the simulated hologram, and the reconstructed image. All frames are compiled into an animated GIF, offering a compelling visualization of wavefront evolution, interference, and inverse imaging under motion. This project demonstrates advanced skills in wave optics, numerical modeling, and scientific animation, and serves as a powerful educational and research tool in computational imaging and digital holography.

## 📌 Overview
This project simulates the formation and reconstruction of digital holograms under dynamic conditions. A grayscale image is treated as a complex object field and undergoes translation, rotation, and scaling over time. Simultaneously, the virtual camera plane oscillates along the optical axis. The result is a time-varying hologram and reconstruction sequence visualized as an animated GIF.

## 🧠 Physics Context
- **Wavefront Propagation**: Fresnel diffraction models near-field propagation.
- **Interference & Holography**: Object and reference waves produce a hologram.
- **Inverse Reconstruction**: Back-propagation recovers the object field.
- **Motion Simulation**: Object and camera motion introduce realistic dynamics.

## 🚀 Features
- Object motion: translation, rotation, zoom.
- Camera motion: axial oscillation.
- Adjustable parameters: grid size, wavelength, reference tilt, noise level.
- Visualizes object, hologram, and reconstruction per frame.
- Saves animation as GIF for presentation or publication.

##🎯 Research & Educational Value
## Ideal for:
• 	Teaching wave optics and holography under motion.

• 	Demonstrating Fresnel propagation and inverse imaging.

• 	Visualizing dynamic interference and reconstruction.
## 👤 Author
Alhassan Kpahambang Fuseini.
## 📄 License
MIT License.

