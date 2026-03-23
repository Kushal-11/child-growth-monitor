document.addEventListener('DOMContentLoaded', function () {

    // ── Image preview helper ─────────────────────────────────────────────
    function wirePreview(inputId, containerId, imgId) {
        const input = document.getElementById(inputId);
        const container = document.getElementById(containerId);
        const img = document.getElementById(imgId);
        if (!input) return;
        input.addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function (evt) {
                    img.src = evt.target.result;
                    container.classList.remove('d-none');
                };
                reader.readAsDataURL(file);
            } else {
                container.classList.add('d-none');
            }
        });
    }

    wirePreview('image',      'imagePreview', 'previewImg');
    wirePreview('image_back', 'backPreview',  'backPreviewImg');
    wirePreview('image_side', 'sidePreview',  'sidePreviewImg');

    // ── Age-in-months ↔ Date-of-birth toggle ────────────────────────────
    const ageMonthsRow   = document.getElementById('ageMonthsRow');
    const dobRow         = document.getElementById('dobRow');
    const ageMonthsInput = document.getElementById('age_months_input');
    const dobPicker      = document.getElementById('dob_picker');
    const dobHidden      = document.getElementById('dob_hidden');
    const toggleDobLink  = document.getElementById('toggleDobLink');
    const toggleAgeLink  = document.getElementById('toggleAgeLink');

    // Convert fractional months to approximate YYYY-MM-DD
    function monthsToDob(months) {
        const ms  = Math.round(months * 30.4375 * 86400 * 1000);
        const dob = new Date(Date.now() - ms);
        return dob.toISOString().slice(0, 10);
    }

    function syncFromAge() {
        const val = parseFloat(ageMonthsInput ? ageMonthsInput.value : '');
        if (dobHidden) {
            dobHidden.value = (!isNaN(val) && val >= 0 && val <= 120)
                ? monthsToDob(val) : '';
        }
    }

    function syncFromDob() {
        if (dobHidden && dobPicker) dobHidden.value = dobPicker.value;
    }

    if (ageMonthsInput) ageMonthsInput.addEventListener('input', syncFromAge);
    if (dobPicker)       dobPicker.addEventListener('change', syncFromDob);

    if (toggleDobLink) {
        toggleDobLink.addEventListener('click', function (e) {
            e.preventDefault();
            ageMonthsRow.classList.add('d-none');
            dobRow.classList.remove('d-none');
            if (ageMonthsInput) ageMonthsInput.value = '';
            if (dobHidden) dobHidden.value = '';
        });
    }

    if (toggleAgeLink) {
        toggleAgeLink.addEventListener('click', function (e) {
            e.preventDefault();
            dobRow.classList.add('d-none');
            ageMonthsRow.classList.remove('d-none');
            if (dobPicker) dobPicker.value = '';
            if (dobHidden) dobHidden.value = '';
        });
    }

    // ── Height unit conversion ───────────────────────────────────────────
    const heightValueInput = document.getElementById('height_value');
    const heightUnitSelect = document.getElementById('height_unit');

    if (heightValueInput && heightUnitSelect) {
        const hiddenHeight = document.createElement('input');
        hiddenHeight.type  = 'hidden';
        hiddenHeight.name  = 'height_cm';
        const form = document.getElementById('assessForm');
        if (form) form.appendChild(hiddenHeight);

        function updateHeightCm() {
            const val = parseFloat(heightValueInput.value);
            hiddenHeight.value = (!isNaN(val) && val > 0)
                ? ((heightUnitSelect.value === 'inch') ? (val * 2.54).toFixed(1) : val.toFixed(1))
                : '';
        }

        heightValueInput.addEventListener('input', updateHeightCm);
        heightUnitSelect.addEventListener('change', function () {
            heightValueInput.max = (heightUnitSelect.value === 'inch') ? 80 : 200;
            updateHeightCm();
        });
    }

    // ── Form submission validation + loading state ───────────────────────
    const form      = document.getElementById('assessForm');
    const submitBtn = document.getElementById('submitBtn');

    if (form && submitBtn) {
        form.addEventListener('submit', function (e) {
            const dob = dobHidden ? dobHidden.value : '';
            if (!dob) {
                e.preventDefault();
                alert("Please enter the child's age (in months) or date of birth.");
                if (ageMonthsInput && ageMonthsRow && !ageMonthsRow.classList.contains('d-none')) {
                    ageMonthsInput.focus();
                } else if (dobPicker) {
                    dobPicker.focus();
                }
                return false;
            }
            submitBtn.disabled = true;
            submitBtn.innerHTML =
                '<span class="spinner-border spinner-border-sm me-2" ' +
                'role="status" aria-hidden="true"></span>Processing…';
        });
    }

});
