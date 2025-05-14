const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const styleSelect = document.getElementById("styleSelect");

// Access the webcam
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((err) => {
      console.error("Error accessing camera: ", err);
      alert("Could not access the camera. Please allow camera permissions.");
    });
} else {
  alert("Your browser does not support webcam access.");
}

// Define styles (paths to pre-generated images)
const styles = {
  style1: "url(/static/images/style1.jpg)",
  style2: "url(/static/images/style2.jpg)",
  style3: "url(/static/images/style3.jpg)",
  style4: "url(/static/images/style4.jpg)",
};

// Function to change the overlay style
function changeStyle() {
  const selectedStyle = styleSelect.value;
  overlay.style.backgroundImage = styles[selectedStyle];
}

// Set initial style
changeStyle();

// Capture the combined image
function capture() {
  const context = canvas.getContext("2d");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw the video frame
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Draw the overlay by temporarily making it opaque
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  const tempContext = tempCanvas.getContext("2d");
  tempContext.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Create an image element for the overlay
  const overlayImage = new Image();
  overlayImage.src = `/static/images/${styleSelect.value}.jpg`;
  overlayImage.onload = () => {
    tempContext.globalAlpha = 0.5; // Match the CSS opacity
    tempContext.drawImage(overlayImage, 0, 0, canvas.width, canvas.height);

    // Draw the combined result back to the main canvas
    context.drawImage(tempCanvas, 0, 0);

    // Download the image
    const imageData = canvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = imageData;
    link.download = "styled_image.png";
    link.click();
  };
}
