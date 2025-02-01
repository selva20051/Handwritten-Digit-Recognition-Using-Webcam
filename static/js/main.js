const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultDiv = document.getElementById("result");

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    video.srcObject = stream;
    video.play();

    // Set canvas size to match video
    canvas.width = 640;
    canvas.height = 480;
  } catch (error) {
    console.error("Error accessing camera:", error);
    resultDiv.innerHTML = "Error: Could not access camera";
  }
}

function sendFrame() {
  // Draw current video frame to canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert canvas to blob
  canvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append("frame", blob);

    fetch("/process_frame", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.processed_frame) {
          const img = new Image();
          img.src = "data:image/jpeg;base64," + data.processed_frame;
          resultDiv.innerHTML = "";
          resultDiv.appendChild(img);
        }
      })
      .catch((error) => console.error("Error:", error));
  }, "image/jpeg");
}

// Start camera and processing
setupCamera().then(() => {
  // Send frame every 100ms
  setInterval(sendFrame, 100);
});