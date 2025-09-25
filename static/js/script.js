function closePopup() {
    const popup = document.getElementById('popup');
    if (popup) {
        popup.style.display = 'none';
    }
}

// Auto close popup after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const popup = document.getElementById('popup');
    if (popup) {
        setTimeout(closePopup, 5000);
    }
});