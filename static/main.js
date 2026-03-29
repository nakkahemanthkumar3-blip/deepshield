const fileInput = document.getElementById('file-input');
const fileNameDiv = document.getElementById('file-name');

if (fileInput) {
  fileInput.addEventListener('change', function () {
    if (this.files && this.files[0]) {
      fileNameDiv.textContent = 'Selected: ' + this.files[0].name;
    }
  });
}