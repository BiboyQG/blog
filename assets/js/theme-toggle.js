// Theme toggle functionality - wait for DOM to be ready
function initThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');

    if (themeToggle) {
        themeToggle.addEventListener('click', function () {
            console.log('Theme toggle clicked'); // Debug log

            if (document.body.classList.contains('dark')) {
                document.body.classList.remove('dark');
                localStorage.setItem('pref-theme', 'light');
                console.log('Switched to light theme'); // Debug log
            } else {
                document.body.classList.add('dark');
                localStorage.setItem('pref-theme', 'dark');
                console.log('Switched to dark theme'); // Debug log
            }
        });
        console.log('Theme toggle initialized successfully'); // Debug log
    } else {
        console.error('Theme toggle button not found!'); // Debug log
    }
}

// Initialize theme on page load - safer approach
function initTheme() {
    if (document.body) {
        const savedTheme = localStorage.getItem('pref-theme');

        if (savedTheme === 'dark') {
            document.body.classList.add('dark');
        } else if (savedTheme === 'light') {
            document.body.classList.remove('dark');
        } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.body.classList.add('dark');
        }
        console.log('Theme initialized'); // Debug log
    }
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
        initTheme();
        initThemeToggle();
    });
} else {
    // DOM is already ready
    initTheme();
    initThemeToggle();
}
