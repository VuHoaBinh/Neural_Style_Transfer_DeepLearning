const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const resultVideo = document.getElementById("resultVideo");
const statusDiv = document.getElementById("status");

uploadBtn.onclick = () => {
  const file = videoInput.files[0];
  if (!file) {
    alert("Please select a video file first.");
    return;
  }

  statusDiv.textContent =
    "Uploading and processing... This may take some time.";

  const formData = new FormData();
  formData.append("video", file);

  fetch("http://127.0.0.1:5000/upload", {
    method: "POST",
    body: formData,
  })
    .then((res) => {
      if (!res.ok) throw new Error("Server error");
      return res.blob();
    })
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      resultVideo.src = url;
      resultVideo.style.display = "block";
      statusDiv.textContent = "Processing complete! Video ready below.";
    })
    .catch((err) => {
      statusDiv.textContent = "Error: " + err.message;
    });
};
