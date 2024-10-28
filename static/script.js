 // Function to scroll to the results section after page load
 window.onload = function() {
    var prediction = "{{ prediction }}";  // Check if a prediction exists
    if (prediction) {
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    }
};

let currentSlide = 0;
const slides = document.querySelectorAll('.hero-slide');
const totalSlides = slides.length;

function showSlide(index) {
    slides.forEach((slide, i) => {
        slide.classList.remove('active');
        if (i === index) {
            slide.classList.add('active');
        }
    });
}

function nextSlide() {
    currentSlide = (currentSlide + 1) % totalSlides;
    showSlide(currentSlide);
}

setInterval(nextSlide, 5000); // Change image every 5 seconds