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


    // Height unit conversion: convert to cm before submission
    const heightValueInput = document.getElementById('height_value');
    const heightUnitSelect = document.getElementById('height_unit');
    if (heightValueInput && heightUnitSelect) {
        // Update max value based on selected unit
        function updateMaxValue() {
            const unit = heightUnitSelect.value;
            if (unit === 'inch') {
                heightValueInput.max = 80; // ~200 cm in inches
                heightValueInput.placeholder = 'Optional (max 80 inches)';
            } else {
                heightValueInput.max = 200;
                heightValueInput.placeholder = 'Optional (max 200 cm)';
            }
        }
        heightUnitSelect.addEventListener('change', updateMaxValue);
        updateMaxValue(); // Set initial max value
        
        // Create hidden input for height_cm
        const hiddenHeightInput = document.createElement('input');
        hiddenHeightInput.type = 'hidden';
        hiddenHeightInput.name = 'height_cm';
        hiddenHeightInput.id = 'height_cm';
        document.getElementById('assessForm').appendChild(hiddenHeightInput);
        
        // Update hidden input when value or unit changes
        function updateHeightCm() {
            const value = parseFloat(heightValueInput.value);
            if (!isNaN(value) && value > 0) {
                const unit = heightUnitSelect.value;
                const heightCm = unit === 'inch' ? value * 2.54 : value;
                hiddenHeightInput.value = heightCm.toFixed(1);
            } else {
                hiddenHeightInput.value = '';
            }
        }
        
        heightValueInput.addEventListener('input', updateHeightCm);
        heightUnitSelect.addEventListener('change', function() {
            updateMaxValue();
            updateHeightCm();
        });
    }

    // Show loading state on form submit
    const form = document.getElementById('assessForm');
    const submitBtn = document.getElementById('submitBtn');

    if (form && submitBtn) {
        form.addEventListener('submit', function (e) {
            // Validate date format before submission
            if (dateInput && dateInput.value) {
                const isoDate = dateInput.getAttribute('data-iso-date');
                if (!isoDate) {
                    e.preventDefault();
                    alert('Please enter a valid date in dd/mm/yyyy format (e.g., 15/01/2022)');
                    dateInput.focus();
                    dateInput.classList.add('is-invalid');
                    return false;
                }
                // Create hidden input with ISO date format for backend
                const hiddenDateInput = document.createElement('input');
                hiddenDateInput.type = 'hidden';
                hiddenDateInput.name = 'date_of_birth';
                hiddenDateInput.value = isoDate;
                form.appendChild(hiddenDateInput);
                // Temporarily disable original input to prevent duplicate submission
                const originalName = dateInput.name;
                dateInput.removeAttribute('name');
            }
            
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        });
    }
});
