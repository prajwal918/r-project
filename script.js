// Tab Navigation
document.addEventListener('DOMContentLoaded', function() {
    // Get all tab links
    const tabLinks = document.querySelectorAll('.tab-link');
    
    // Add click event listener to each tab link
    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get the target tab ID
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all tab links
            tabLinks.forEach(link => link.classList.remove('active'));
            
            // Add active class to clicked tab link
            this.classList.add('active');
            
            // Hide all tab content
            const allTabContent = document.querySelectorAll('.tab-content article');
            allTabContent.forEach(article => article.classList.remove('active'));
            
            // Show the target tab content
            const targetArticle = document.getElementById(targetTab);
            if (targetArticle) {
                targetArticle.classList.add('active');
            }
        });
    });
    
    // Toggle switches functionality
    const toggleSwitches = document.querySelectorAll('.toggle-switch');
    
    toggleSwitches.forEach(toggle => {
        toggle.addEventListener('click', function() {
            this.classList.toggle('active');
        });
    });
    
    // Danger buttons with confirmation
    const deactivateBtn = document.querySelector('.danger-btn:nth-of-type(1)');
    const deleteBtn = document.querySelector('.danger-btn:nth-of-type(2)');
    
    if (deactivateBtn) {
        deactivateBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to deactivate your account? You can reactivate it later.')) {
                alert('Account deactivation would be processed here.');
            }
        });
    }
    
    if (deleteBtn) {
        deleteBtn.addEventListener('click', function() {
            const confirmation = prompt('This action cannot be undone. Type "DELETE" to confirm:');
            if (confirmation === 'DELETE') {
                alert('Account deletion would be processed here.');
            } else if (confirmation !== null) {
                alert('Deletion cancelled. Text did not match.');
            }
        });
    }
    
    // Follow button functionality
    const followBtn = document.querySelector('.profile-actions .primary-btn:nth-of-type(2)');
    if (followBtn) {
        followBtn.addEventListener('click', function() {
            if (this.textContent.includes('Follow')) {
                this.textContent = '✓ Following';
                this.style.background = '#00c853';
                this.style.borderColor = '#00c853';
            } else {
                this.textContent = '➕ Follow';
                this.style.background = '#0095f6';
                this.style.borderColor = '#0095f6';
            }
        });
    }
    
    // Message button functionality
    const messageBtn = document.querySelector('.profile-actions button:nth-of-type(3)');
    if (messageBtn) {
        messageBtn.addEventListener('click', function() {
            alert('Message functionality would open a chat window here.');
        });
    }
    
    // Share button functionality
    const shareBtn = document.querySelector('.profile-actions button:nth-of-type(4)');
    if (shareBtn) {
        shareBtn.addEventListener('click', function() {
            if (navigator.share) {
                navigator.share({
                    title: 'u/username - SocialHub Profile',
                    text: 'Check out this profile on SocialHub!',
                    url: window.location.href
                }).catch(err => console.log('Error sharing:', err));
            } else {
                // Fallback for browsers that don't support Web Share API
                const url = window.location.href;
                navigator.clipboard.writeText(url).then(() => {
                    alert('Profile link copied to clipboard!');
                }).catch(() => {
                    alert('Share this profile: ' + url);
                });
            }
        });
    }
    
    // Settings button functionality
    const settingsBtn = document.querySelector('.profile-actions button:last-of-type');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', function() {
            // Click on the settings tab
            const settingsTab = document.querySelector('[data-tab="settings"]');
            if (settingsTab) {
                settingsTab.click();
                // Scroll to settings
                window.scrollTo({
                    top: document.querySelector('.tabs-navigation').offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    }
    
    // Navigation buttons
    const homeBtn = document.querySelector('.nav-btn:nth-of-type(1)');
    const exploreBtn = document.querySelector('.nav-btn:nth-of-type(2)');
    const messagesBtn = document.querySelector('.nav-btn:nth-of-type(3)');
    const createBtn = document.querySelector('.nav-btn.primary');
    
    if (homeBtn) {
        homeBtn.addEventListener('click', () => {
            alert('Home page would load here');
        });
    }
    
    if (exploreBtn) {
        exploreBtn.addEventListener('click', () => {
            alert('Explore page would load here');
        });
    }
    
    if (messagesBtn) {
        messagesBtn.addEventListener('click', () => {
            alert('Messages page would load here');
        });
    }
    
    if (createBtn) {
        createBtn.addEventListener('click', () => {
            alert('Create post dialog would open here');
        });
    }
    
    // Achievement badges hover effect with title
    const achievementBadges = document.querySelectorAll('.achievement-badge');
    const achievementTitles = [
        '🎂 Cake Day',
        '✉️ First Message',
        '📝 First Post',
        '⭐ Super Contributor',
        '💬 Comment Champion',
        '🎁 Gift Giver'
    ];
    
    achievementBadges.forEach((badge, index) => {
        badge.title = achievementTitles[index] || 'Achievement';
        badge.style.cursor = 'pointer';
    });
});
