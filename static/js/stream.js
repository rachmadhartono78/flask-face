const video = document.getElementById('localVideo');

// Access local camera
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(stream => {
        video.srcObject = stream;

        // Send video frames to server
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        function sendFrame() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert frame to blob
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('frame', blob);

                // POST frame to Flask server
                fetch('/receive_frame', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                })
                .catch(err => console.error(err));
            }, 'image/jpeg');

            requestAnimationFrame(sendFrame);
        }

        video.onloadedmetadata = () => {
            sendFrame();
        };
    })
    .catch(err => console.error('Error accessing camera:', err));
