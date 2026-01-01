async function predict() {
    const fileInput = document.getElementById("fileInput");
    const resultText = document.getElementById("prediction");

    // Check if a file is selected
    if (fileInput.files.length === 0) {
        alert("Please select an image file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Send the image to the backend API
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', { // Flask API URL
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Display the result
        if (data.prediction) {
            resultText.innerText = `Predicted Disease: ${data.prediction}`;
        } else if (data.error) {
            resultText.innerText = `Error: ${data.error}`;
        }
    } catch (error) {
        resultText.innerText = `Error: ${error.message}`;
    }
}
