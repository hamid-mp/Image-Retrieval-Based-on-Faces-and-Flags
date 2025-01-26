$(function() {
    // Vars.
    var $window = $(window),
        $body = $('body'),
        $wrapper = $('#wrapper');

    // Breakpoints.
    skel.breakpoints({
        xlarge: '(max-width: 1680px)',
        large: '(max-width: 1280px)',
        medium: '(max-width: 980px)',
        small: '(max-width: 736px)',
        xsmall: '(max-width: 480px)'
    });

    // Disable animations/transitions until everything's loaded.
    $body.addClass('is-loading');

    $window.on('load', function() {
        $body.removeClass('is-loading');
    });

    // Poptrox.
    $window.on('load', function() {
        $('.thumbnails').poptrox({
            onPopupClose: function() { $body.removeClass('is-covered'); },
            onPopupOpen: function() { $body.addClass('is-covered'); },
            baseZIndex: 10001,
            useBodyOverflow: false,
            usePopupEasyClose: true,
            overlayColor: '#000000',
            overlayOpacity: 0.75,
            popupLoaderText: '',
            fadeSpeed: 500,
            usePopupDefaultStyling: false,
            windowMargin: (skel.breakpoint('small').active? 5 : 50)
        });
    });

    // Image Upload and Display
    $(document).on('change', '#image-upload', function(event) {
        const file = event.target.files[0];
        if (file) {
            console.log('File selected:', file);
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const uploadedImage = $('#uploaded-image');
                    uploadedImage.attr('src', e.target.result);
                    console.log('Image source set to:', e.target.result);
                    uploadedImage.show(); // Ensure the image is shown
                    $('#upload-label').hide(); // Hide the upload label

                    // Now send the image to the backend for inference
                    sendImageToBackend(file);
                } catch (error) {
                    console.error('Error parsing uploaded image:', error);
                }
            };
            reader.readAsDataURL(file);
        } else {
            console.log('No file selected');
        }
    });

    $(document).on('click', '#uploaded-image', function() {
        $('#image-upload').click();
    });

    // Function to send the image to the backend
    function sendImageToBackend(file) {
        const formData = new FormData();
        formData.append('image', file);
    
        fetch('http://127.0.0.1:5000/upload-image', { 
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            console.log('YOU HAVE RECIEVED THE BACKEND RESPONSE')
            return response.json(); // Parse the JSON response
        })
        .then(data => {
            console.log('Received data:', data);  // Log the data to verify
        
            // Now handle the data (if it's valid JSON)
            if (data && typeof data === 'object') {
                const resultText = Object.entries(data).map(([key, value]) => `${key}: ${value}`).join('\n');
                $('#result-content').text(resultText);
                $('#result-container').show();

            } else {
                console.error('Unexpected response format:', data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            const resultContainer = $('#result-container');
            const resultContent = $('#result-content');
            resultContent.text(`Error: ${error.message}`);
            resultContainer.show();
        });        
    }
});