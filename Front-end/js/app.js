const uploadContainer = document.getElementById('uploadContainer');
const popup = document.getElementById('popup');
const overlay = document.getElementById('overlay');
const closePopup = document.getElementById('closePopup');
const errorMessage = document.getElementById('errorMessage');
const process_btn = document.getElementById('process_btn');
const cancel_btn = document.getElementById('cancel_btn');
const download_btn = document.getElementById('download_btn');
let processing_start = false;

uploadContainer.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadContainer.classList.add('dragover');
});

uploadContainer.addEventListener('dragleave', () => {
  uploadContainer.classList.remove('dragover');
});

uploadContainer.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadContainer.classList.remove('dragover');

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    const file = files[0];
    if (!file.type.startsWith('video/')) {
      showErrorPopup('Invalid file type. Please upload a video file.');
    } else {
      uploadFile(file)
        .then((response) => showErrorPopup(`File "${file.name}" uploaded successfully.`))
        .catch((error) => showErrorPopup('Error uploading file: ' + error.message));
    }
  }
});

const uploadFile = (file) => {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    })
      .then((response) => {
        if (response.ok) {
            process_btn.classList.remove('disabled');
            cancel_btn.classList.remove('disabled');

            add_event_process_btn();
            add_event_cancel_btn();

            return response.json();
        } else {
          throw new Error('Failed to upload file.');
        }
      })
      .then((data) => resolve(data))
      .catch((error) => reject(error));
  });
};

function showErrorPopup(message) {
  errorMessage.textContent = message;
  popup.classList.add('active');
  overlay.classList.add('active');
}

closePopup.addEventListener('click', (e) => {
  e.preventDefault();
  popup.classList.remove('active');
  overlay.classList.remove('active');
  if (processing_start) {
    // toggleLoader();
  }
});

overlay.addEventListener('click', (e) => {
  e.preventDefault();
  popup.classList.remove('active');
  overlay.classList.remove('active');
});


const callProcessVideo = async () => {
    toggleLoader();
    try {
        const response = await fetch('http://localhost:5000/process', {
            method: 'POST',
        });

        if (!response.ok) {
            throw new Error('Failed to process video.');
        }

        const data = await response.json();
        toggleLoader();
        download_btn.classList.remove('disabled');
        add_event_download_btn();  
        showErrorPopup(data.message);
    } catch (error) {
        toggleLoader();
        showErrorPopup('Error processing video: ' + error.message);
    }
};



const add_event_process_btn = () => {
    process_btn.addEventListener('click', async () => {
      await callProcessVideo();
      processing_start = true;
    });
  };

const add_event_cancel_btn = () => {
    cancel_btn.addEventListener('click', () => {
        window.location.reload();
    });
};

const add_event_download_btn = () => {
        download_btn.addEventListener('click', () => {
            // Make a POST request to the /download endpoint
            fetch('http://127.0.0.1:5000/download', {
                method: 'POST',
            })
            .then(response => {
                if (response.ok) {
                    return response.blob(); // Get the file blob if the response is successful
                }
                throw new Error('File not found');
            })
            .then(blob => {
                // Create a temporary link to download the file
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'output.mp4'; // The name of the file being downloaded
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            });
        });        
};



const overlay_ = document.getElementById('overlay');
const toggleLoaderBtn = document.getElementById('toggleLoader');

const toggleLoader = () => {
    overlay_.style.display = (overlay_.style.display === 'none' || overlay_.style.display === '') ? 'flex' : 'none';
};
