// Find Image Groups - Client-side JavaScript

let clusters = [];
let ungroupedImages = [];
let currentCluster = 0;
let availableColors = [];
let focusedImageIndex = 0;
let showUngrouped = false;
let gridBrightness = 100; // Global brightness for grid view
let currentThreshold = 0.85; // Current similarity threshold
let isReclustering = false; // Flag to prevent multiple simultaneous re-clustering
let eyeDetectionEnabled = false; // Track if eye detection is enabled

// Color filter state (updated logic)
let colorFilterMode = 'all';          // 'all' or 'selective'
let selectedColorFilters = new Set(); // Colors to SHOW (not hide)

// Eye detection filter state
let closedEyesFilterActive = false;  // true = show only closed eyes

// View mode state
let viewMode = 'groups';  // 'groups' or 'browse'

// Browse All mode state
let allImages = [];                    // All images for Browse All
let allImagesLoaded = false;           // Track if all-images data has been loaded
let browseCurrentPage = 0;             // Current page in Browse All
let browseImagesPerPage = 20;          // Images per page (10/20/50/all)
let browseSortBy = 'date';             // 'name' or 'date'
let browseSortAscending = true;        // Sort direction
let browseFilteredImages = [];         // After applying color filter

// State preservation for mode toggling
let savedGroupsState = {
    currentCluster: 0,
    focusedImageIndex: 0
};
let savedBrowseState = {
    currentPage: 0,
    imagesPerPage: 20,
    sortBy: 'date',
    sortAscending: true
};

// Lightbox state
let lightboxViewMode = 'groups'; // 'groups' or 'browse' - track which mode lightbox was opened from
let lightboxOpen = false;
let lightboxImageIndex = 0;
let zoomLevel = 1;
let brightnessLevel = 100;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let panOffsetX = 0;
let panOffsetY = 0;

// DOM elements
const loading = document.getElementById('loading');
const viewer = document.getElementById('viewer');
const errorDiv = document.getElementById('error');
const imageGrid = document.getElementById('imageGrid');
const similarityList = document.getElementById('similarityList');
const groupInfo = document.getElementById('groupInfo');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

// Initialize
async function init() {
    try {
        // Load only what's needed for Groups mode initially (faster startup)
        // Browse mode data will be loaded on-demand when user switches to Browse
        const [colorsResponse, clustersResponse, ungroupedResponse, configResponse] = await Promise.all([
            fetch('/api/colors'),
            fetch('/api/clusters'),
            fetch('/api/ungrouped'),
            fetch('/api/config')
        ]);

        if (!colorsResponse.ok || !clustersResponse.ok || !ungroupedResponse.ok || !configResponse.ok) {
            throw new Error('Failed to load data');
        }

        availableColors = await colorsResponse.json();
        clusters = await clustersResponse.json();
        ungroupedImages = await ungroupedResponse.json();
        // allImages will be loaded lazily when switching to Browse mode

        const config = await configResponse.json();

        // Set initial UI state from config
        currentThreshold = config.threshold;
        eyeDetectionEnabled = config.eye_detection_enabled || false;

        // Debug logging
        console.log('Eye detection enabled:', eyeDetectionEnabled);

        // Set the radio button for the current threshold
        const thresholdRadios = document.querySelectorAll('input[name="threshold"]');
        thresholdRadios.forEach(radio => {
            if (parseFloat(radio.value) === config.threshold) {
                radio.checked = true;
            }
        });

        // Set the direct-only checkbox
        const directOnlyCheckbox = document.getElementById('directOnlyCheckbox');
        if (directOnlyCheckbox) {
            directOnlyCheckbox.checked = config.direct_only;
        }

        // Show closed eyes filter button if eye detection is enabled
        const closedEyesFilterBtn = document.getElementById('closedEyesFilterBtn');
        if (closedEyesFilterBtn) {
            closedEyesFilterBtn.style.display = eyeDetectionEnabled ? '' : 'none';
            console.log('Closed eyes filter button visible:', eyeDetectionEnabled);
        }

        showUngrouped = ungroupedImages.length > 0;

        if (clusters.length === 0 && !showUngrouped) {
            showError();
            return;
        }

        loading.classList.add('hidden');
        viewer.classList.remove('hidden');

        showCluster(0);
        createMainColorFilter(); // Initialize main color filter
        setupEventListeners();
    } catch (error) {
        console.error('Error loading clusters:', error);
        loading.classList.add('hidden');
        errorDiv.classList.remove('hidden');
        errorDiv.querySelector('p').textContent = `Error: ${error.message}`;
    }
}

// Show error state
function showError() {
    loading.classList.add('hidden');
    errorDiv.classList.remove('hidden');
}

// Display a cluster
// Create eye detection badge for an image
function createEyeBadge(eyeDetection) {
    if (!eyeDetectionEnabled || !eyeDetection) {
        return null;
    }

    const badge = document.createElement('div');
    badge.className = 'eye-badge';

    const status = eyeDetection.status;
    let icon = '';
    let title = '';
    let badgeClass = '';

    if (status === 'open') {
        icon = '👁️';
        title = `Eyes Open (score: ${eyeDetection.score.toFixed(3)})`;
        badgeClass = 'eye-open';
    } else if (status === 'closed') {
        icon = '😑';
        title = `Eyes Closed (score: ${eyeDetection.score.toFixed(3)})`;
        badgeClass = 'eye-closed';
    } else if (status === 'no_face') {
        icon = '❓';
        title = 'No Face Detected';
        badgeClass = 'eye-no-face';
    } else if (status === 'error') {
        icon = '⚠';
        title = 'Eye Detection Error';
        badgeClass = 'eye-error';
    }

    badge.className += ` ${badgeClass}`;
    badge.textContent = icon;
    badge.title = title;

    return badge;
}

function showCluster(index) {
    // Handle ungrouped images as a special "cluster"
    if (showUngrouped && index >= clusters.length) {
        showUngroupedImages();
        return;
    }

    // Wrap around
    currentCluster = ((index % clusters.length) + clusters.length) % clusters.length;

    const cluster = clusters[currentCluster];

    // Filter images if closed eyes filter is active
    let imagesToShow = cluster.images;
    if (eyeDetectionEnabled && closedEyesFilterActive) {
        imagesToShow = cluster.images.filter(img => {
            return img.eye_detection && img.eye_detection.status === 'closed';
        });
    }

    // Update header info
    const totalImages = cluster.num_images;
    const showingImages = imagesToShow.length;
    if (closedEyesFilterActive && showingImages !== totalImages) {
        groupInfo.textContent = `Group ${currentCluster + 1} of ${clusters.length} (showing ${showingImages} of ${totalImages} images)`;
    } else {
        groupInfo.textContent = `Group ${currentCluster + 1} of ${clusters.length} (${totalImages} images)`;
    }

    // Clear previous content
    imageGrid.innerHTML = '';
    similarityList.innerHTML = '';

    // Reset focused image
    focusedImageIndex = 0;

    // Create image cards
    imagesToShow.forEach((image, idx) => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.dataset.imageIndex = idx;

        // Set initial focus
        if (idx === 0) {
            card.classList.add('focused');
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'image-wrapper';
        wrapper.style.cursor = 'pointer';
        wrapper.title = 'Click to zoom';

        // Click to open lightbox
        wrapper.addEventListener('click', () => {
            openLightbox(idx);
        });

        const img = document.createElement('img');
        img.className = 'loading';
        img.alt = image.filename;
        img.src = `/api/image/${currentCluster}/${idx}`;
        img.style.filter = `brightness(${gridBrightness}%)`;

        img.onload = () => {
            img.classList.remove('loading');
        };

        img.onerror = () => {
            img.alt = 'Failed to load';
            img.classList.remove('loading');
        };

        wrapper.appendChild(img);

        // Add eye detection badge if available
        if (eyeDetectionEnabled && image.eye_detection) {
            const eyeBadge = createEyeBadge(image.eye_detection);
            if (eyeBadge) {
                wrapper.appendChild(eyeBadge);
            }
        }

        const filename = document.createElement('div');
        filename.className = 'image-filename';
        filename.textContent = image.filename;

        // Create color picker
        const colorPicker = createColorPicker(image, idx);

        card.appendChild(wrapper);
        card.appendChild(filename);
        card.appendChild(colorPicker);
        imageGrid.appendChild(card);
    });

    // Create similarity items
    cluster.similarities.forEach(sim => {
        const item = document.createElement('div');
        item.className = 'similarity-item';

        const files = document.createElement('div');
        files.className = 'similarity-files';
        files.innerHTML = `
            <span>${sim.img1}</span>
            <span class="similarity-arrow">↔</span>
            <span>${sim.img2}</span>
        `;

        const stats = document.createElement('div');
        stats.innerHTML = `
            <span class="similarity-percentage">Similarity: ${sim.percentage.toFixed(1)}%</span>
            <span class="similarity-score">(score: ${sim.similarity.toFixed(4)})</span>
        `;

        item.appendChild(files);
        item.appendChild(stats);
        similarityList.appendChild(item);
    });

    // Apply color filter if active (must be done AFTER cards are created)
    if (selectedColorFilters.size > 0) {
        applyGroupsColorFilter();
    }

    // Update button states
    updateButtons();
}

// Display ungrouped images
function showUngroupedImages() {
    currentCluster = clusters.length; // Special index for ungrouped

    // Filter images if closed eyes filter is active
    let imagesToShow = ungroupedImages;
    if (eyeDetectionEnabled && closedEyesFilterActive) {
        imagesToShow = ungroupedImages.filter(img => {
            return img.eye_detection && img.eye_detection.status === 'closed';
        });
    }

    // Update header info
    const totalImages = ungroupedImages.length;
    const showingImages = imagesToShow.length;
    if (closedEyesFilterActive && showingImages !== totalImages) {
        groupInfo.textContent = `Ungrouped Images (showing ${showingImages} of ${totalImages} images)`;
    } else {
        groupInfo.textContent = `Ungrouped Images (${totalImages} images)`;
    }

    // Clear previous content
    imageGrid.innerHTML = '';
    similarityList.innerHTML = '';

    // Add info message
    const infoDiv = document.createElement('div');
    infoDiv.className = 'ungrouped-info';
    infoDiv.innerHTML = '<p>These images have no similar counterparts based on the current threshold.</p>';
    similarityList.appendChild(infoDiv);

    // Reset focused image
    focusedImageIndex = 0;

    // Create image cards
    imagesToShow.forEach((image, idx) => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.dataset.imageIndex = idx;

        // Set initial focus
        if (idx === 0) {
            card.classList.add('focused');
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'image-wrapper';
        wrapper.style.cursor = 'pointer';
        wrapper.title = 'Click to zoom';

        // Click to open lightbox
        wrapper.addEventListener('click', () => {
            openLightbox(idx);
        });

        const img = document.createElement('img');
        img.className = 'loading';
        img.alt = image.filename;
        img.src = `/api/ungrouped/${idx}`;
        img.style.filter = `brightness(${gridBrightness}%)`;

        img.onload = () => {
            img.classList.remove('loading');
        };

        img.onerror = () => {
            img.alt = 'Failed to load';
            img.classList.remove('loading');
        };

        wrapper.appendChild(img);

        // Add eye detection badge if available
        if (eyeDetectionEnabled && image.eye_detection) {
            const eyeBadge = createEyeBadge(image.eye_detection);
            if (eyeBadge) {
                wrapper.appendChild(eyeBadge);
            }
        }

        const filename = document.createElement('div');
        filename.className = 'image-filename';
        filename.textContent = image.filename;

        // Create color picker
        const colorPicker = createColorPicker(image, idx);

        card.appendChild(wrapper);
        card.appendChild(filename);
        card.appendChild(colorPicker);
        imageGrid.appendChild(card);
    });

    // Apply color filter if active (must be done AFTER cards are created)
    if (selectedColorFilters.size > 0) {
        applyGroupsColorFilter();
    }

    // Update button states
    updateButtons();
}

// Create color picker for an image
function createColorPicker(image, imageIdx) {
    const picker = document.createElement('div');
    picker.className = 'color-picker';

    const label = document.createElement('span');
    label.className = 'color-picker-label';
    label.textContent = 'Tag:';
    picker.appendChild(label);

    availableColors.forEach(color => {
        const btn = document.createElement('button');
        btn.className = 'color-btn';
        btn.setAttribute('data-color', color);
        btn.title = color;

        // Set selected state
        if (image.color === color || (color === 'None' && !image.color)) {
            btn.classList.add('selected');
        }

        // Click handler
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            await setImageColor(imageIdx, color, picker);
        });

        picker.appendChild(btn);
    });

    return picker;
}

// Set color for an image
async function setImageColor(imageIdx, color, pickerElement) {
    try {
        let response;
        const isUngrouped = currentCluster >= clusters.length;
        
        if (isUngrouped) {
            response = await fetch(`/api/ungrouped/color/${imageIdx}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ color })
            });
        } else {
            response = await fetch(`/api/color/${currentCluster}/${imageIdx}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ color })
            });
        }

        if (!response.ok) {
            throw new Error('Failed to set color');
        }

        const result = await response.json();

        // Update UI
        const buttons = pickerElement.querySelectorAll('.color-btn');
        buttons.forEach(btn => {
            if (btn.getAttribute('data-color') === color) {
                btn.classList.add('selected');
            } else {
                btn.classList.remove('selected');
            }
        });

        // Update data
        if (isUngrouped) {
            ungroupedImages[imageIdx].color = color === 'None' ? null : color;
        } else {
            clusters[currentCluster].images[imageIdx].color = color === 'None' ? null : color;
        }

    } catch (error) {
        console.error('Error setting color:', error);
        alert('Failed to set color tag. Check console for details.');
    }
}

// Update navigation buttons
function updateButtons() {
    prevBtn.disabled = false;
    nextBtn.disabled = false;
}

// Navigate to next cluster
function nextCluster() {
    const totalGroups = getTotalGroups();
    const nextIndex = currentCluster + 1;
    if (nextIndex >= totalGroups) {
        showCluster(0); // Wrap around to first group
    } else {
        showCluster(nextIndex);
    }
}

// Navigate to previous cluster
function prevCluster() {
    const totalGroups = getTotalGroups();
    const prevIndex = currentCluster - 1;
    if (prevIndex < 0) {
        showCluster(totalGroups - 1); // Wrap around to last group
    } else {
        showCluster(prevIndex);
    }
}

// Get total number of groups including ungrouped
function getTotalGroups() {
    let total = clusters.length;
    if (showUngrouped && ungroupedImages.length > 0) {
        total += 1;
    }
    return total;
}

// ===== MODE TOGGLE FUNCTIONS =====

function switchToGroupsMode() {
    if (viewMode === 'browse') {
        // Save Browse state
        savedBrowseState = {
            currentPage: browseCurrentPage,
            imagesPerPage: browseImagesPerPage,
            sortBy: browseSortBy,
            sortAscending: browseSortAscending
        };
    }

    viewMode = 'groups';

    // Update UI
    document.getElementById('groupsModeBtn').classList.add('active');
    document.getElementById('browseModeBtn').classList.remove('active');
    document.getElementById('groupsNav').style.display = '';
    document.getElementById('browseNav').style.display = 'none';
    document.getElementById('groupsControls').style.display = '';
    document.getElementById('browseControls').style.display = 'none';

    // Show "Show Details" button in groups mode
    const toggleSimilaritiesBtn = document.getElementById('toggleSimilarities');
    if (toggleSimilaritiesBtn) {
        toggleSimilaritiesBtn.style.display = '';
    }

    // Restore Groups state
    currentCluster = savedGroupsState.currentCluster;
    focusedImageIndex = savedGroupsState.focusedImageIndex;

    // Show current cluster
    showCluster(currentCluster);
}

async function switchToBrowseMode() {
    if (viewMode === 'groups') {
        // Save Groups state
        savedGroupsState = {
            currentCluster: currentCluster,
            focusedImageIndex: focusedImageIndex
        };
    }

    viewMode = 'browse';

    // Update UI
    document.getElementById('browseModeBtn').classList.add('active');
    document.getElementById('groupsModeBtn').classList.remove('active');
    document.getElementById('browseNav').style.display = '';
    document.getElementById('groupsNav').style.display = 'none';
    document.getElementById('browseControls').style.display = '';
    document.getElementById('groupsControls').style.display = 'none';

    // Hide "Show Details" button in browse mode (no similarities to show)
    const toggleSimilaritiesBtn = document.getElementById('toggleSimilarities');
    if (toggleSimilaritiesBtn) {
        toggleSimilaritiesBtn.style.display = 'none';
    }

    // Load all-images data if not already loaded (lazy loading for performance)
    if (!allImagesLoaded) {
        // Show loading indicator
        imageGrid.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading all images data...</p></div>';

        try {
            const allImagesResponse = await fetch('/api/all-images');
            if (!allImagesResponse.ok) {
                throw new Error('Failed to load all images');
            }

            allImages = await allImagesResponse.json();

            // Add index to each image for API calls
            allImages.forEach((img, idx) => {
                img.index = idx;
            });

            allImagesLoaded = true;
        } catch (error) {
            console.error('Error loading all images:', error);
            imageGrid.innerHTML = '<div class="error"><p>Failed to load images data.</p></div>';
            return;
        }
    }

    // Restore Browse state
    browseCurrentPage = savedBrowseState.currentPage;
    browseImagesPerPage = savedBrowseState.imagesPerPage;
    browseSortBy = savedBrowseState.sortBy;
    browseSortAscending = savedBrowseState.sortAscending;

    // Update UI controls
    document.getElementById('imagesPerPageSelect').value =
        browseImagesPerPage === -1 ? 'all' : browseImagesPerPage;
    document.getElementById('sortBySelect').value = browseSortBy;
    updateSortOrderButton();

    // Show current page
    showBrowsePage(browseCurrentPage);
}

// ===== BROWSE ALL MODE FUNCTIONS =====

function sortAllImages() {
    // Safety check: if allImages is empty or not loaded, return empty array
    if (!allImages || allImages.length === 0) {
        console.warn('sortAllImages called but allImages is empty');
        return [];
    }

    let sorted = [...allImages];

    if (browseSortBy === 'name') {
        sorted.sort((a, b) => a.filename.localeCompare(b.filename));
    } else if (browseSortBy === 'date') {
        // Sort by EXIF creation date
        sorted.sort((a, b) => {
            return a.creation_date.localeCompare(b.creation_date);
        });
    }

    if (!browseSortAscending) {
        sorted.reverse();
    }

    return sorted;
}

function applyBrowseFilters() {
    const sorted = sortAllImages();

    let filtered = sorted;

    // Apply color filter (only if colors are actually selected)
    if (colorFilterMode === 'selective' && selectedColorFilters.size > 0) {
        filtered = filtered.filter(img => {
            const color = img.color || 'None';
            return selectedColorFilters.has(color);
        });
    }

    // Apply closed eyes filter (if active)
    if (eyeDetectionEnabled && closedEyesFilterActive) {
        filtered = filtered.filter(img => {
            return img.eye_detection && img.eye_detection.status === 'closed';
        });
    }

    browseFilteredImages = filtered;
}

function showBrowsePage(pageIndex) {
    applyBrowseFilters();

    const totalImages = browseFilteredImages.length;

    // Debug logging
    console.log('showBrowsePage called:', {
        pageIndex,
        totalImages,
        allImagesLength: allImages.length,
        allImagesLoaded
    });

    const perPage = browseImagesPerPage === -1 ? totalImages : browseImagesPerPage;
    const totalPages = perPage === 0 ? 1 : Math.ceil(totalImages / perPage);

    // Clamp page index
    browseCurrentPage = Math.max(0, Math.min(pageIndex, totalPages - 1));

    // Calculate slice
    const startIdx = browseCurrentPage * perPage;
    const endIdx = browseImagesPerPage === -1 ? totalImages : startIdx + perPage;
    const pageImages = browseFilteredImages.slice(startIdx, endIdx);

    // Update header
    document.getElementById('browsePageInfo').textContent =
        `Page ${browseCurrentPage + 1} of ${totalPages} (${pageImages.length}/${totalImages} images)`;

    // Render grid
    imageGrid.innerHTML = '';

    if (pageImages.length === 0) {
        console.warn('No images to display');
        imageGrid.innerHTML = '<div class="error"><p>No images to display</p></div>';
        return;
    }

    pageImages.forEach((img, idx) => {
        const globalIdx = startIdx + idx;
        const card = createBrowseImageCard(img, globalIdx);
        imageGrid.appendChild(card);
    });

    // Reset focus (browse mode doesn't use focus highlighting)
    focusedImageIndex = 0;

    // Hide similarities section in browse mode
    if (similarityList) {
        similarityList.classList.remove('visible');
    }
}

function createBrowseImageCard(img, displayIdx) {
    const card = document.createElement('div');
    card.className = 'image-card';
    card.dataset.imageIndex = displayIdx;

    // Image wrapper with click for lightbox
    const wrapper = document.createElement('div');
    wrapper.className = 'image-wrapper';
    wrapper.style.cursor = 'pointer';
    wrapper.title = 'Click to zoom';

    wrapper.addEventListener('click', () => {
        openBrowseLightbox(displayIdx);  // Use display index for lightbox
    });

    // Image element
    const imgElem = document.createElement('img');
    imgElem.src = `/api/all-images/${img.index}`;  // Use original index
    imgElem.alt = img.filename;
    imgElem.loading = 'lazy';
    imgElem.style.filter = `brightness(${gridBrightness}%)`;

    wrapper.appendChild(imgElem);

    // Add eye detection badge if available
    if (eyeDetectionEnabled && img.eye_detection) {
        const eyeBadge = createEyeBadge(img.eye_detection);
        if (eyeBadge) {
            wrapper.appendChild(eyeBadge);
        }
    }

    // Filename
    const filename = document.createElement('div');
    filename.className = 'image-filename';
    filename.textContent = img.filename;

    // Create color picker manually for browse mode
    const colorPicker = document.createElement('div');
    colorPicker.className = 'color-picker';

    const label = document.createElement('span');
    label.className = 'color-picker-label';
    label.textContent = 'Tag:';
    colorPicker.appendChild(label);

    availableColors.forEach(color => {
        const btn = document.createElement('button');
        btn.className = 'color-btn';
        btn.setAttribute('data-color', color);
        btn.title = color;

        // Set selected state
        if (img.color === color || (color === 'None' && !img.color)) {
            btn.classList.add('selected');
        }

        // Click handler for browse mode
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();  // Prevent lightbox from opening

            const response = await fetch(`/api/all-images/color/${img.index}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ color: color })
            });

            if (response.ok) {
                // Update UI
                const buttons = colorPicker.querySelectorAll('.color-btn');
                buttons.forEach(b => {
                    if (b.getAttribute('data-color') === color) {
                        b.classList.add('selected');
                    } else {
                        b.classList.remove('selected');
                    }
                });

                // Update data
                img.color = color === 'None' ? null : color;
                allImages[img.index].color = color === 'None' ? null : color;

                // Refresh if filter active
                if (colorFilterMode === 'selective') {
                    showBrowsePage(browseCurrentPage);
                }
            }
        });

        colorPicker.appendChild(btn);
    });

    card.appendChild(wrapper);
    card.appendChild(filename);
    card.appendChild(colorPicker);

    return card;
}

function nextBrowsePage() {
    const totalImages = browseFilteredImages.length;
    const perPage = browseImagesPerPage === -1 ? totalImages : browseImagesPerPage;
    const totalPages = perPage === 0 ? 1 : Math.ceil(totalImages / perPage);

    browseCurrentPage = (browseCurrentPage + 1) % totalPages;
    showBrowsePage(browseCurrentPage);
}

function prevBrowsePage() {
    const totalImages = browseFilteredImages.length;
    const perPage = browseImagesPerPage === -1 ? totalImages : browseImagesPerPage;
    const totalPages = perPage === 0 ? 1 : Math.ceil(totalImages / perPage);

    browseCurrentPage = (browseCurrentPage - 1 + totalPages) % totalPages;
    showBrowsePage(browseCurrentPage);
}

function updateSortOrderButton() {
    const btn = document.getElementById('sortOrderBtn');
    if (btn) {
        btn.textContent = browseSortAscending ? '↑ Ascending' : '↓ Descending';
    }
}

// ===== BROWSE ALL LIGHTBOX FUNCTIONS =====

function openBrowseLightbox(imageIndex) {
    lightboxOpen = true;
    lightboxImageIndex = imageIndex;
    lightboxViewMode = 'browse';
    zoomLevel = 1;
    brightnessLevel = 100;
    panOffsetX = 0;
    panOffsetY = 0;

    const lightbox = document.getElementById('lightbox');
    lightbox.classList.remove('hidden');

    showBrowseLightboxImage();
    setupLightboxEventListeners();
}

async function showBrowseLightboxImage() {
    const img = browseFilteredImages[lightboxImageIndex];

    // Update image
    const lightboxImg = document.getElementById('lightboxImage');
    lightboxImg.src = `/api/all-images/${img.index}`;  // Use original index

    // Update filename (use class selector since HTML doesn't have ID)
    const lightboxFilename = document.querySelector('.lightbox-filename');
    if (lightboxFilename) {
        lightboxFilename.textContent = img.filename;
    }

    // Update image counter
    const imageNumElem = document.getElementById('lightboxImageNum');
    if (imageNumElem) {
        imageNumElem.textContent = `${lightboxImageIndex + 1} of ${browseFilteredImages.length}`;
    }

    // Fetch and display EXIF
    try {
        const exifResponse = await fetch(`/api/all-images/exif/${img.index}`);  // Use original index
        if (exifResponse.ok) {
            const exif = await exifResponse.json();
            const exifHeaderElem = document.getElementById('lightboxExifHeader');
            const exifFooterElem = document.getElementById('lightboxExif');  // Correct ID
            if (exifHeaderElem && exifFooterElem) {
                displayExifData(exif, exifHeaderElem, exifFooterElem, img.eye_detection);
            }
        }
    } catch (err) {
        console.error('Error loading EXIF:', err);
    }

    // Update color picker in lightbox
    updateLightboxColorPicker(img.color, img.index);  // Use original index

    // Update image with zoom and brightness
    lightboxImg.style.transform = `scale(${zoomLevel}) translate(${panOffsetX}px, ${panOffsetY}px)`;
    lightboxImg.style.filter = `brightness(${brightnessLevel}%)`;

    // Update zoom and brightness displays
    const zoomDisplay = document.querySelector('.lightbox-zoom-level');
    const brightnessDisplay = document.querySelector('.lightbox-brightness-level');
    if (zoomDisplay) zoomDisplay.textContent = `${Math.round(zoomLevel * 100)}%`;
    if (brightnessDisplay) brightnessDisplay.textContent = `${brightnessLevel}%`;
}

function updateLightboxColorPicker(currentColor, originalImageIdx) {
    const lightboxColorPicker = document.getElementById('lightboxColorPicker');
    if (!lightboxColorPicker) return;

    // Clear existing picker
    lightboxColorPicker.innerHTML = '';

    // Create color picker manually for browse mode lightbox
    const picker = document.createElement('div');
    picker.className = 'color-picker';

    const label = document.createElement('span');
    label.className = 'color-picker-label';
    label.textContent = 'Tag:';
    picker.appendChild(label);

    availableColors.forEach(color => {
        const btn = document.createElement('button');
        btn.className = 'color-btn';
        btn.setAttribute('data-color', color);
        btn.title = color;

        // Set selected state
        if (currentColor === color || (color === 'None' && !currentColor)) {
            btn.classList.add('selected');
        }

        // Click handler for browse mode lightbox
        btn.addEventListener('click', async (e) => {
            e.preventDefault();

            const response = await fetch(`/api/all-images/color/${originalImageIdx}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ color: color })
            });

            if (response.ok) {
                // Update allImages array
                allImages[originalImageIdx].color = color === 'None' ? null : color;
                // Update current image in browseFilteredImages
                browseFilteredImages[lightboxImageIndex].color = color === 'None' ? null : color;
                // Update picker display
                updateLightboxColorPicker(color === 'None' ? null : color, originalImageIdx);
            }
        });

        picker.appendChild(btn);
    });

    lightboxColorPicker.appendChild(picker);
}

// Set focus to an image
function focusImage(index) {
    const isUngrouped = currentCluster >= clusters.length;
    const totalImages = isUngrouped ? ungroupedImages.length : clusters[currentCluster].images.length;

    // Wrap around
    focusedImageIndex = ((index % totalImages) + totalImages) % totalImages;

    // Update UI
    const cards = document.querySelectorAll('.image-card');
    cards.forEach((card, idx) => {
        if (idx === focusedImageIndex) {
            card.classList.add('focused');
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            card.classList.remove('focused');
        }
    });
}

// Set grid brightness
function setGridBrightness(newBrightness) {
    gridBrightness = Math.max(20, Math.min(200, newBrightness));
    
    // Update all images in the current view
    const images = document.querySelectorAll('.image-wrapper img');
    images.forEach(img => {
        img.style.filter = `brightness(${gridBrightness}%)`;
    });
    
    // Update brightness display if in grid view
    if (!lightboxOpen) {
        const brightnessDisplay = document.getElementById('gridBrightnessDisplay');
        if (brightnessDisplay) {
            brightnessDisplay.textContent = `${gridBrightness}%`;
        }
    }
}

// Reset grid brightness
function resetGridBrightness() {
    setGridBrightness(100);
}

// Threshold control functions
function getSelectedThreshold() {
    const selected = document.querySelector('input[name="threshold"]:checked');
    return selected ? parseFloat(selected.value) : 0.85;
}

function updateThresholdStatus(message, isError = false) {
    const status = document.getElementById('thresholdStatus');
    if (status) {
        status.textContent = message;
        status.style.color = isError ? '#f44336' : '#4CAF50';
    }
}

// Color filter functions
function createColorFilterUI() {
    const filterContainer = document.getElementById('lightboxColorFilter');
    if (!filterContainer) {
        console.log('Color filter container not found');
        return;
    }
    
    filterContainer.innerHTML = '';
    
    // Create color filter buttons for each color (except "None")
    const filterableColors = availableColors.filter(color => color !== 'None');
    
    console.log('Available colors for filtering:', filterableColors);
    
    filterableColors.forEach(color => {
        const btn = document.createElement('button');
        btn.className = 'filter-color-btn';
        btn.title = `Hide ${color} images`;
        btn.dataset.color = color;
        
        // Set color based on the color name
        const colorMap = {
            'Red': '#f44336',
            'Orange': '#ff9800',
            'Yellow': '#ffeb3b',
            'Green': '#4CAF50',
            'Blue': '#2196F3',
            'Purple': '#9c27b0',
            'Pink': '#e91e63'
        };
        
        btn.style.backgroundColor = colorMap[color] || '#999';
        
        // Check if this color is currently filtered
        if (activeColorFilters.has(color)) {
            btn.classList.add('active');
        }
        
        btn.addEventListener('click', () => toggleColorFilter(color, btn));
        filterContainer.appendChild(btn);
    });
}

function toggleColorFilter(color, button) {
    if (activeColorFilters.has(color)) {
        activeColorFilters.delete(color);
        button.classList.remove('active');
    } else {
        activeColorFilters.add(color);
        button.classList.add('active');
    }
    
    // Apply the filter to the current view
    applyColorFilter();
}

function clearColorFilter() {
    activeColorFilters.clear();
    const buttons = document.querySelectorAll('.filter-color-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    applyMainColorFilter();
    updateFilterStatus();
}

// Main color filter functions
function createMainColorFilter() {
    const filterContainer = document.getElementById('mainColorFilter');
    if (!filterContainer) {
        console.log('Main color filter container not found');
        return;
    }

    filterContainer.innerHTML = '';

    // Create color filter buttons for each color (except "None")
    const filterableColors = availableColors.filter(color => color !== 'None');

    console.log('Creating main color filter with colors:', filterableColors);

    filterableColors.forEach(color => {
        const btn = document.createElement('button');
        btn.className = 'filter-color-btn';
        btn.title = `Show only ${color}`;
        btn.dataset.color = color;

        // Set color based on the color name
        const colorMap = {
            'Red': '#f44336',
            'Orange': '#ff9800',
            'Yellow': '#ffeb3b',
            'Green': '#4CAF50',
            'Blue': '#2196F3',
            'Purple': '#9c27b0',
            'Pink': '#e91e63'
        };

        btn.style.backgroundColor = colorMap[color] || '#999';

        btn.addEventListener('click', () => toggleSelectiveColorFilter(color));
        filterContainer.appendChild(btn);
    });

    updateColorFilterButtons();
    updateFilterStatus();
}

function toggleSelectiveColorFilter(color) {
    // Deactivate "Show All"
    colorFilterMode = 'selective';
    const showAllBtn = document.getElementById('showAllFilterBtn');
    if (showAllBtn) {
        showAllBtn.classList.remove('active');
    }

    // Toggle color selection
    if (selectedColorFilters.has(color)) {
        selectedColorFilters.delete(color);
    } else {
        selectedColorFilters.add(color);
    }

    // If no colors selected AND closed eyes filter is not active, revert to "Show All"
    if (selectedColorFilters.size === 0 && !closedEyesFilterActive) {
        activateShowAll();
        return;
    }

    // Update button states
    updateColorFilterButtons();

    // Apply filter
    applyColorFilter();
}

function activateShowAll() {
    colorFilterMode = 'all';
    selectedColorFilters.clear();
    closedEyesFilterActive = false; // Also reset closed eyes filter

    const showAllBtn = document.getElementById('showAllFilterBtn');
    if (showAllBtn) {
        showAllBtn.classList.add('active');
    }

    // Reset closed eyes filter button
    const closedEyesBtn = document.getElementById('closedEyesFilterBtn');
    if (closedEyesBtn) {
        closedEyesBtn.classList.remove('active');
    }

    updateColorFilterButtons();
    applyColorFilter();
}

function toggleClosedEyesFilter() {
    if (!eyeDetectionEnabled) return;

    closedEyesFilterActive = !closedEyesFilterActive;

    const closedEyesBtn = document.getElementById('closedEyesFilterBtn');
    const showAllBtn = document.getElementById('showAllFilterBtn');

    if (closedEyesFilterActive) {
        // Activate closed eyes filter
        closedEyesBtn.classList.add('active');
        // Deactivate "Show All" since we're filtering now
        if (showAllBtn) {
            showAllBtn.classList.remove('active');
        }
        // Don't change colorFilterMode - let it stay as is
    } else {
        // Deactivate closed eyes filter
        closedEyesBtn.classList.remove('active');
        // If no color filters are active, switch back to Show All
        if (selectedColorFilters.size === 0 && colorFilterMode === 'selective') {
            colorFilterMode = 'all';
            if (showAllBtn) {
                showAllBtn.classList.add('active');
            }
        }
    }

    // Apply filter based on current view mode
    if (viewMode === 'browse') {
        browseCurrentPage = 0; // Reset to first page
        showBrowsePage(0);
    } else {
        // In groups mode, refresh current cluster
        showCluster(currentCluster);
    }

    updateFilterStatus();
}

function updateColorFilterButtons() {
    const buttons = document.querySelectorAll('.filter-color-btn');
    buttons.forEach(btn => {
        const color = btn.getAttribute('data-color');
        if (selectedColorFilters.has(color)) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

function updateFilterStatus() {
    const status = document.getElementById('filterStatus');
    if (!status) return;

    const filters = [];

    // Add color filters
    if (selectedColorFilters.size > 0) {
        filters.push(Array.from(selectedColorFilters).join(', '));
    }

    // Add closed eyes filter
    if (closedEyesFilterActive) {
        filters.push('Closed Eyes');
    }

    if (filters.length === 0) {
        status.textContent = 'Showing all images';
        status.style.color = '#4CAF50';
    } else {
        const count = viewMode === 'groups'
            ? document.querySelectorAll('.image-card').length
            : browseFilteredImages.length;
        status.textContent = `Showing ${filters.join(' + ')} (${count} images)`;
        status.style.color = '#FF9800';
    }
}

function applyColorFilter() {
    if (viewMode === 'groups') {
        applyGroupsColorFilter();
    } else {
        // Re-render browse page with new filter
        showBrowsePage(browseCurrentPage);
    }

    updateFilterStatus();
}

function applyGroupsColorFilter() {
    const imageCards = document.querySelectorAll('.image-card');

    if (selectedColorFilters.size === 0) {
        // Show all images
        imageCards.forEach(card => {
            card.style.display = '';
        });
    } else {
        // Show only selected colors
        imageCards.forEach(card => {
            const colorPicker = card.querySelector('.color-picker');
            if (!colorPicker) {
                card.style.display = 'none';
                return;
            }

            const selectedBtn = colorPicker.querySelector('.color-btn.selected');
            const imageColor = selectedBtn ? selectedBtn.getAttribute('data-color') : 'None';

            if (selectedColorFilters.has(imageColor)) {
                card.style.display = '';
            } else {
                card.style.display = 'none';
            }
        });
    }
}


async function applyNewThreshold() {
    if (isReclustering) {
        updateThresholdStatus('Re-clustering already in progress...', true);
        return;
    }

    // Get the selected threshold value from radio buttons
    const selectedThreshold = getSelectedThreshold();
    currentThreshold = selectedThreshold;

    // Get the direct-only checkbox state
    const directOnlyCheckbox = document.getElementById('directOnlyCheckbox');
    const directOnly = directOnlyCheckbox ? directOnlyCheckbox.checked : false;

    console.log('Applying new threshold:', currentThreshold, 'direct only:', directOnly);

    isReclustering = true;
    updateThresholdStatus('Re-clustering...', false);

    try {
        const response = await fetch('/api/recluster', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                threshold: currentThreshold,
                direct_only: directOnly
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Re-clustering failed');
        }
        
        const result = await response.json();

        console.log('Re-cluster result:', result);
        console.log('New clusters:', result.clusters.length);
        console.log('Stats:', result.stats);

        // Update clusters and ungrouped images
        clusters = result.clusters;
        ungroupedImages = result.ungrouped;
        showUngrouped = ungroupedImages.length > 0;

        // Reset to first cluster
        currentCluster = 0;
        focusedImageIndex = 0;

        // Update display
        showCluster(0);

        // Update status
        const stats = result.stats;
        updateThresholdStatus(`Success! ${stats.num_clusters} groups, ${stats.num_ungrouped} ungrouped`, false);
        
    } catch (error) {
        console.error('Re-clustering error:', error);
        updateThresholdStatus(`Error: ${error.message}`, true);
    } finally {
        isReclustering = false;
    }
}

// Tag focused image with color
async function tagFocusedImage(colorIndex) {
    if (colorIndex < 0 || colorIndex >= availableColors.length) return;

    const color = availableColors[colorIndex];
    const isUngrouped = currentCluster >= clusters.length;
    
    if (!isUngrouped) {
        const cluster = clusters[currentCluster];
        if (!cluster) return;
    }

    const card = document.querySelector(`.image-card[data-image-index="${focusedImageIndex}"]`);
    if (!card) return;

    const picker = card.querySelector('.color-picker');
    await setImageColor(focusedImageIndex, color, picker);

    // Auto-advance to next image
    focusImage(focusedImageIndex + 1);
}

// Set up event listeners
function setupEventListeners() {
    // Mode toggle buttons
    const groupsModeBtn = document.getElementById('groupsModeBtn');
    const browseModeBtn = document.getElementById('browseModeBtn');
    if (groupsModeBtn) {
        groupsModeBtn.addEventListener('click', switchToGroupsMode);
    }
    if (browseModeBtn) {
        browseModeBtn.addEventListener('click', switchToBrowseMode);
    }

    // Groups mode navigation
    prevBtn.addEventListener('click', prevCluster);
    nextBtn.addEventListener('click', nextCluster);

    // Browse mode navigation
    const browsePrevBtn = document.getElementById('browsePrevBtn');
    const browseNextBtn = document.getElementById('browseNextBtn');
    if (browsePrevBtn) {
        browsePrevBtn.addEventListener('click', prevBrowsePage);
    }
    if (browseNextBtn) {
        browseNextBtn.addEventListener('click', nextBrowsePage);
    }

    // Browse controls
    const imagesPerPageSelect = document.getElementById('imagesPerPageSelect');
    if (imagesPerPageSelect) {
        imagesPerPageSelect.addEventListener('change', (e) => {
            browseImagesPerPage = e.target.value === 'all' ? -1 : parseInt(e.target.value);
            browseCurrentPage = 0; // Reset to first page
            showBrowsePage(0);
        });
    }

    const sortBySelect = document.getElementById('sortBySelect');
    if (sortBySelect) {
        sortBySelect.addEventListener('change', (e) => {
            browseSortBy = e.target.value;
            browseCurrentPage = 0;
            showBrowsePage(0);
        });
    }

    const sortOrderBtn = document.getElementById('sortOrderBtn');
    if (sortOrderBtn) {
        sortOrderBtn.addEventListener('click', () => {
            browseSortAscending = !browseSortAscending;
            updateSortOrderButton();
            browseCurrentPage = 0;
            showBrowsePage(0);
        });
    }

    // Closed eyes filter button
    const closedEyesFilterBtn = document.getElementById('closedEyesFilterBtn');
    if (closedEyesFilterBtn) {
        closedEyesFilterBtn.addEventListener('click', toggleClosedEyesFilter);
    }

    // Show All filter button
    const showAllFilterBtn = document.getElementById('showAllFilterBtn');
    if (showAllFilterBtn) {
        showAllFilterBtn.addEventListener('click', activateShowAll);
    }
    
    // Set up grid brightness button event listeners
    const brightnessDownBtn = document.getElementById('gridBrightnessDown');
    const brightnessUpBtn = document.getElementById('gridBrightnessUp');
    const brightnessResetBtn = document.getElementById('gridBrightnessReset');
    
    if (brightnessDownBtn) {
        brightnessDownBtn.onclick = () => setGridBrightness(gridBrightness - 10);
    }
    if (brightnessUpBtn) {
        brightnessUpBtn.onclick = () => setGridBrightness(gridBrightness + 10);
    }
    if (brightnessResetBtn) {
        brightnessResetBtn.onclick = resetGridBrightness;
    }
    
    // Set up threshold control event listeners
    const thresholdApplyBtn = document.getElementById('thresholdApply');
    if (thresholdApplyBtn) {
        thresholdApplyBtn.onclick = applyNewThreshold;
    }

    // Set up radio button listeners to update current threshold when changed
    const thresholdRadios = document.querySelectorAll('input[name="threshold"]');
    thresholdRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            currentThreshold = parseFloat(radio.value);
        });
    });
    
    // Set up color filter clear buttons
    const clearFilterBtn = document.getElementById('clearColorFilter');
    if (clearFilterBtn) {
        clearFilterBtn.onclick = clearColorFilter;
    }
    
    const clearMainFilterBtn = document.getElementById('clearMainColorFilter');
    if (clearMainFilterBtn) {
        clearMainFilterBtn.onclick = () => {
            clearColorFilter();
            updateFilterStatus();
        };
    }

    // Toggle similarities button
    const toggleSimilaritiesBtn = document.getElementById('toggleSimilarities');
    if (toggleSimilaritiesBtn) {
        toggleSimilaritiesBtn.onclick = () => {
            const similaritiesDiv = document.getElementById('similarities');
            if (similaritiesDiv) {
                similaritiesDiv.classList.toggle('visible');

                // Update button text and style
                if (similaritiesDiv.classList.contains('visible')) {
                    toggleSimilaritiesBtn.textContent = 'Hide Details';
                    toggleSimilaritiesBtn.classList.add('active');
                } else {
                    toggleSimilaritiesBtn.textContent = 'Show Details';
                    toggleSimilaritiesBtn.classList.remove('active');
                }
            }
        };
    }

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        // Don't interfere with text input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        // Lightbox controls
        if (lightboxOpen) {
            switch(e.key) {
                case 'Escape':
                    e.preventDefault();
                    closeLightbox();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    if (e.shiftKey || e.ctrlKey) {
                        // Pan left when zoomed
                        if (zoomLevel > 1) {
                            panOffsetX += 50;
                            showLightboxImage();
                        }
                    } else {
                        lightboxPrevImage();
                    }
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    if (e.shiftKey || e.ctrlKey) {
                        // Pan right when zoomed
                        if (zoomLevel > 1) {
                            panOffsetX -= 50;
                            showLightboxImage();
                        }
                    } else {
                        lightboxNextImage();
                    }
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    if (zoomLevel > 1) {
                        panOffsetY += 50;
                        showLightboxImage();
                    }
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    if (zoomLevel > 1) {
                        panOffsetY -= 50;
                        showLightboxImage();
                    }
                    break;
                case '+':
                case '=':
                    e.preventDefault();
                    setZoom(zoomLevel * 1.2);
                    break;
                case '-':
                case '_':
                    e.preventDefault();
                    setZoom(zoomLevel / 1.2);
                    break;
                case ' ':
                    e.preventDefault();
                    fitToScreen();
                    break;
                case 'y':
                case 'Y':
                    e.preventDefault();
                    setBrightness(brightnessLevel - 10);
                    break;
                case 'x':
                case 'X':
                    e.preventDefault();
                    setBrightness(brightnessLevel + 10);
                    break;
                case ',':
                case '<':
                case ';':  // German keyboard: Shift + ,
                case ':':  // German keyboard: Shift + .
                    e.preventDefault();
                    setBrightness(brightnessLevel - 10);
                    break;
                case '.':
                case '>':
                case ':':  // German keyboard: Shift + .
                case ';':  // German keyboard: Shift + ,
                    e.preventDefault();
                    setBrightness(brightnessLevel + 10);
                    break;
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                    e.preventDefault();
                    const colorIndex = parseInt(e.key) - 1;
                    if (colorIndex >= 0 && colorIndex < availableColors.length) {
                        const color = availableColors[colorIndex];
                        const picker = document.querySelector('#lightboxColorPicker .color-picker');
                        setImageColor(lightboxImageIndex, color, picker);
                    }
                    break;
            }
            return;
        }

        // Normal grid view controls
        switch(e.key) {
            case 'ArrowLeft':
            case 'a':
            case 'A':
            case 'p':
            case 'P':
                e.preventDefault();
                if (viewMode === 'browse') {
                    prevBrowsePage();
                } else {
                    prevCluster();
                }
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
            case 'n':
            case 'N':
                e.preventDefault();
                if (viewMode === 'browse') {
                    nextBrowsePage();
                } else {
                    nextCluster();
                }
                break;
            case 'q':
            case 'Q':
            case 'Escape':
                e.preventDefault();
                if (confirm('Close the viewer?')) {
                    window.close();
                }
                break;
            case 'Tab':
                e.preventDefault();
                if (e.shiftKey) {
                    focusImage(focusedImageIndex - 1);
                } else {
                    focusImage(focusedImageIndex + 1);
                }
                break;
            case 'y':
            case 'Y':
                e.preventDefault();
                setGridBrightness(gridBrightness - 10);
                break;
            case 'x':
            case 'X':
                e.preventDefault();
                setGridBrightness(gridBrightness + 10);
                break;
            case ' ':
                e.preventDefault();
                resetGridBrightness();
                break;
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
                e.preventDefault();
                const colorIndex = parseInt(e.key) - 1;
                tagFocusedImage(colorIndex);
                break;
        }
    });
}

// Lightbox functions
function openLightbox(imageIndex) {
    lightboxOpen = true;
    lightboxImageIndex = imageIndex;
    lightboxViewMode = 'groups'; // Set to groups mode
    zoomLevel = 1;
    brightnessLevel = 100;
    panOffsetX = 0;
    panOffsetY = 0;

    const lightbox = document.getElementById('lightbox');
    lightbox.classList.remove('hidden');

    console.log('Opening lightbox, available colors:', availableColors);
    showLightboxImage();
    createColorFilterUI(); // Initialize color filter UI
    setupLightboxEventListeners();
}

function closeLightbox() {
    lightboxOpen = false;
    const lightbox = document.getElementById('lightbox');
    lightbox.classList.add('hidden');
}

async function showLightboxImage() {
    let image, cluster;
    const isUngrouped = currentCluster >= clusters.length;
    
    if (isUngrouped) {
        image = ungroupedImages[lightboxImageIndex];
    } else {
        cluster = clusters[currentCluster];
        if (!cluster) return;
        image = cluster.images[lightboxImageIndex];
    }

    const lightboxImg = document.getElementById('lightboxImage');
    const filenameElem = document.querySelector('.lightbox-filename');
    const imageNumElem = document.getElementById('lightboxImageNum');
    const colorPickerElem = document.getElementById('lightboxColorPicker');
    const exifElemHeader = document.getElementById('lightboxExifHeader');
    const exifElemFooter = document.getElementById('lightboxExif');

    // Update image
    if (isUngrouped) {
        lightboxImg.src = `/api/ungrouped/${lightboxImageIndex}`;
    } else {
        lightboxImg.src = `/api/image/${currentCluster}/${lightboxImageIndex}`;
    }
    lightboxImg.style.transform = `scale(${zoomLevel}) translate(${panOffsetX}px, ${panOffsetY}px)`;
    lightboxImg.style.filter = `brightness(${brightnessLevel}%)`;

    // Update filename
    filenameElem.textContent = image.filename;

    // Update image number
    const totalImages = isUngrouped ? ungroupedImages.length : cluster.images.length;
    imageNumElem.textContent = `${lightboxImageIndex + 1} of ${totalImages}`;

    // Update zoom level display
    document.querySelector('.lightbox-zoom-level').textContent = `${Math.round(zoomLevel * 100)}%`;

    // Update brightness level display
    document.querySelector('.lightbox-brightness-level').textContent = `${brightnessLevel}%`;

    // Update color picker
    colorPickerElem.innerHTML = '';
    const picker = createColorPicker(image, lightboxImageIndex);
    picker.classList.remove('color-picker');
    picker.className = 'color-picker';
    colorPickerElem.appendChild(picker);

    // Load and display EXIF data
    try {
        let exifResponse;
        if (isUngrouped) {
            exifResponse = await fetch(`/api/ungrouped/exif/${lightboxImageIndex}`);
        } else {
            exifResponse = await fetch(`/api/exif/${currentCluster}/${lightboxImageIndex}`);
        }
        
        if (exifResponse.ok) {
            const exif = await exifResponse.json();
            console.log('EXIF data received:', exif); // Debug logging
            displayExifData(exif, exifElemHeader, exifElemFooter, image.eye_detection);
        } else {
            console.error('EXIF request failed:', exifResponse.status);
            exifElemHeader.innerHTML = '<span style="color: #888;">No EXIF data available</span>';
        }
    } catch (error) {
        console.error('Error loading EXIF data:', error);
        exifElemHeader.innerHTML = '<span style="color: #888;">Error loading EXIF</span>';
    }
}

function displayExifData(exif, containerHeader, containerFooter, eyeDetection = null) {
    containerHeader.innerHTML = '';
    if (containerFooter) containerFooter.innerHTML = '';

    const fields = [
        { key: 'iso', label: 'ISO' },
        { key: 'shutter_speed', label: 'Shutter' },
        { key: 'aperture', label: 'Aperture' },
        { key: 'focal_length', label: 'Focal' },
        { key: 'exposure_bias', label: 'Exp Comp' }
    ];

    let hasData = false;

    fields.forEach(field => {
        if (exif[field.key]) {
            hasData = true;
            const item = document.createElement('div');
            item.className = 'exif-item';

            const label = document.createElement('div');
            label.className = 'exif-label';
            label.textContent = field.label;

            const value = document.createElement('div');
            value.className = 'exif-value';
            value.textContent = exif[field.key];

            item.appendChild(label);
            item.appendChild(value);
            containerHeader.appendChild(item);
        }
    });

    // Add eye detection info if available
    if (eyeDetectionEnabled && eyeDetection && containerFooter) {
        const eyeItem = document.createElement('div');
        eyeItem.className = 'exif-item eye-detection-exif';

        const label = document.createElement('div');
        label.className = 'exif-label';
        label.textContent = 'Eyes';

        const value = document.createElement('div');
        value.className = 'exif-value';

        const status = eyeDetection.status;
        let statusText = '';
        let statusClass = '';

        if (status === 'open') {
            statusText = '👁️ Open';
            statusClass = 'eye-status-open';
        } else if (status === 'closed') {
            statusText = '😑 Closed';
            statusClass = 'eye-status-closed';
        } else if (status === 'no_face') {
            statusText = '❓ No Face';
            statusClass = 'eye-status-no-face';
        } else if (status === 'error') {
            statusText = '⚠ Error';
            statusClass = 'eye-status-error';
        }

        value.innerHTML = `<span class="${statusClass}">${statusText}</span>`;
        value.title = `Score: ${eyeDetection.score?.toFixed(3)}, Method: ${eyeDetection.method}`;

        eyeItem.appendChild(label);
        eyeItem.appendChild(value);
        containerFooter.appendChild(eyeItem);
    }

    if (!hasData) {
        console.log('No EXIF fields found in data:', exif);
        containerHeader.innerHTML = '<span style="color: #ff9800; font-size: 0.9rem;">⚠ No EXIF data found in file</span>';
    }
}

function setZoom(newZoom) {
    zoomLevel = Math.max(0.5, Math.min(5, newZoom));

    // Reset pan if zoomed out
    if (zoomLevel <= 1) {
        panOffsetX = 0;
        panOffsetY = 0;
    }

    showLightboxImage();
}

function fitToScreen() {
    zoomLevel = 1;
    brightnessLevel = 100;
    panOffsetX = 0;
    panOffsetY = 0;
    showLightboxImage();
}

function setBrightness(newBrightness) {
    brightnessLevel = Math.max(20, Math.min(200, newBrightness));
    const lightboxImg = document.getElementById('lightboxImage');
    lightboxImg.style.filter = `brightness(${brightnessLevel}%)`;
    document.querySelector('.lightbox-brightness-level').textContent = `${brightnessLevel}%`;
}

function lightboxPrevImage() {
    if (lightboxViewMode === 'browse') {
        // Navigate through ALL filtered images, across pages
        const totalImages = browseFilteredImages.length;
        lightboxImageIndex = (lightboxImageIndex - 1 + totalImages) % totalImages;
        showBrowseLightboxImage();
    } else {
        // Groups mode navigation
        const isUngrouped = currentCluster >= clusters.length;
        const totalImages = isUngrouped ? ungroupedImages.length : clusters[currentCluster].images.length;

        lightboxImageIndex = (lightboxImageIndex - 1 + totalImages) % totalImages;
        showLightboxImage();
    }

    // Reset zoom/pan
    zoomLevel = 1;
    brightnessLevel = 100;
    panOffsetX = 0;
    panOffsetY = 0;
}

function lightboxNextImage() {
    if (lightboxViewMode === 'browse') {
        // Navigate through ALL filtered images, across pages
        const totalImages = browseFilteredImages.length;
        lightboxImageIndex = (lightboxImageIndex + 1) % totalImages;
        showBrowseLightboxImage();
    } else {
        // Groups mode navigation
        const isUngrouped = currentCluster >= clusters.length;
        const totalImages = isUngrouped ? ungroupedImages.length : clusters[currentCluster].images.length;

        lightboxImageIndex = (lightboxImageIndex + 1) % totalImages;
        showLightboxImage();
    }

    // Reset zoom/pan
    zoomLevel = 1;
    brightnessLevel = 100;
    panOffsetX = 0;
    panOffsetY = 0;
}

function setupLightboxEventListeners() {
    const lightboxImg = document.getElementById('lightboxImage');
    const container = document.querySelector('.lightbox-image-container');

    // Brightness buttons
    document.getElementById('brightnessUp').onclick = () => setBrightness(brightnessLevel + 10);
    document.getElementById('brightnessDown').onclick = () => setBrightness(brightnessLevel - 10);

    // Zoom buttons
    document.getElementById('zoomIn').onclick = () => setZoom(zoomLevel * 1.2);
    document.getElementById('zoomOut').onclick = () => setZoom(zoomLevel / 1.2);
    document.getElementById('fitScreen').onclick = fitToScreen;
    document.getElementById('closeLightbox').onclick = closeLightbox;

    // Navigation buttons
    document.getElementById('lightboxPrev').onclick = lightboxPrevImage;
    document.getElementById('lightboxNext').onclick = lightboxNextImage;

    // Mouse wheel zoom
    container.onwheel = (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        setZoom(zoomLevel * delta);
    };

    // Pan functionality
    container.onmousedown = (e) => {
        if (zoomLevel > 1) {
            isPanning = true;
            panStartX = e.clientX - panOffsetX;
            panStartY = e.clientY - panOffsetY;
            container.classList.add('grabbing');
        }
    };

    container.onmousemove = (e) => {
        if (isPanning) {
            panOffsetX = e.clientX - panStartX;
            panOffsetY = e.clientY - panStartY;
            lightboxImg.style.transform = `scale(${zoomLevel}) translate(${panOffsetX}px, ${panOffsetY}px)`;
        }
    };

    container.onmouseup = () => {
        isPanning = false;
        container.classList.remove('grabbing');
    };

    container.onmouseleave = () => {
        isPanning = false;
        container.classList.remove('grabbing');
    };
}

// Start the application
init();
