# Instant photo sorter

This project aims to classify a large collection of cellphone images into four specific categories. It's designed for individuals looking to organize and categorize images stored in their devices, particularly those accumulated through apps like WhatsApp. The application utilizes a fine-tuned ResNet50 model for image classification and offers two main functionalities: training your own model with custom data or using the pre-trained model to classify your images.

## Application Screenshots

Here are some screenshots showcasing the application's features and user interface:

![Main Interface](/images/01-main_interface.png "Main Interface of Instant Photo Sorter")

![Classification Process](/images/02-classification_process.png "Classifying Images")

![Results Overview](/images/03-results_overview.png "Overview of Classification Results")

## Features

- Custom model training with user-provided image data.
- Image classification into four predefined categories using a fine-tuned ResNet50 model.
- A GUI for easy interaction with the application for classification tasks.
- Command-line support for model training and customization.
- TensorBoard integration for training visualization and monitoring.
- Detailed evaluation metrics, including classification reports and confusion matrices.

## Getting Started

### Prerequisites

Before installing the Instant Photo Sorter, ensure you have the following prerequisites installed on your system (the requirements.txt file contains the versions used in this project):

- Python 3.8 or newer
- PyTorch 1.8+
- torchvision 0.9+
- PIL (Pillow) 10.2.0
- scikit-learn 1.4.0
- PySide2 5.15.2.1 (for the GUI)
- TensorBoard 2.15.1 (for logging)

### Installation

1. Clone the repository to your local machine:
   ```sh
   git clone https://github.com/aitorporcel/instant-photo-sorter
   ```
2. Navigate to the project directory and install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Training Your Own Model

To train your own model with custom datasets, follow these steps:

1. **Prepare Your Dataset**: Organize your images into folders named after the categories you wish to classify. For example, if categorizing images into 'Memes', 'Landscapes', 'People', and 'Various', each category should have its own folder containing the relevant images. Move all these folders into a new folder named train_data and move some of them into test_data for evaluation.

2. **Train the Model**: Use the command-line interface to start the training process with your dataset. You can customize training parameters such as the number of epochs, batch size, and train/validation split ratio:

   ```sh
   python main.py --num_epochs 30 --batch_size 32 --train_split 0.8
   ```

3. **Monitor Training Progress**: TensorBoard logs training metrics like loss and accuracy. To view these metrics, run TensorBoard and navigate to the provided URL in your web browser:

   ```sh
   tensorboard --logdir runs/
   ```

4. **Evaluate the Model**: After training, the model is automatically evaluated on the validation and test sets. Results, including the classification report and confusion matrix, are displayed in the terminal.

### Classifying Images with the Pre-trained or Custom Model

To classify your images using the GUI:

1. **Start the Application**: Run `app.py` without any arguments to launch the GUI.

   ```sh
   python app.py
   ```

2. **Select Source and Output Folders**: In the GUI, choose the folder containing the images you wish to classify and specify the output directory where classified images will be stored.

3. **Classify Images**: Click the 'Classify Images' button to start the classification process. Classified images will be copied to subfolders in the output directory, organized by category.

## Project Structure

- `main.py`: The application's entry point.
- `utils.py`: Contains utility functions for training and evaluation.
- `train_data/`: Default directory for the training dataset.
- `test_data/`: Default directory for the test dataset.
- `models/`: Directory where trained models are saved.

## Creating an Executable File

You can package this application into an executable file for easier distribution and use, without needing to set up a Python environment. This process is facilitated by PyInstaller, which bundles the application and all its dependencies into a single executable file.

### Prerequisites

- Ensure PyInstaller is installed in your environment:
  ```sh
  pip install pyinstaller
  ```

### Building the Executable

1. **Navigate to Your Project Directory**: Open a terminal or command prompt and change to the directory containing your project files.

2. **Run PyInstaller**: Use the following command to generate the executable.

   ```sh
   pyinstaller --onefile --windowed --add-data="models/model_trained.pth:models" app.py
   ```

   Note: Specify the resource with a source path and a target path, separated by a platform-specific character (`;` for Windows, `:` for Linux/MacOS).

3. **Locate the Executable**: After the build process completes, find your executable in the `dist` directory within your project folder.

4. **Run the Executable**: You can now distribute and run the generated executable file without needing a Python environment installed.

Note: Building an executable with PyInstaller works best when your project is fully functional in a Python environment. If you encounter issues, ensure all dependencies are correctly installed and the application runs as expected before packaging it with PyInstaller.

## Contributing

Contributions are welcome! Please feel free to fork the project, make changes, and submit pull requests with new features or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

LinkedIn: [https://www.linkedin.com/in/aitorporcellaburu/](https://www.linkedin.com/in/aitorporcellaburu/)

Project Link: [https://github.com/aitorporcel/instant-photo-sorter](https://github.com/aitorporcel/instant-photo-sorter)
