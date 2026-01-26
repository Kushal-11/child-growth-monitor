// Image preview on file selection
document.addEventListener('DOMContentLoaded', function () {
    const imageInput = document.getElementById('image');
    const previewContainer = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');

    if (imageInput) {
        imageInput.addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function (evt) {
                    previewImg.src = evt.target.result;
                    previewContainer.classList.remove('d-none');
                };
                reader.readAsDataURL(file);
            } else {
                previewContainer.classList.add('d-none');
            }
        });
    }

    // Show loading state on form submit
    const form = document.getElementById('assessForm');
    const submitBtn = document.getElementById('submitBtn');

    if (form && submitBtn) {
        form.addEventListener('submit', function () {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        });
    }
});
