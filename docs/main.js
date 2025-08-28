// main.js - interactive bits (separated from HTML)
//
// - updates year
// - handles navbar scroll toggle
// - reveals about container when it enters viewport
// - smooth scroll for anchor links
// Keep selectors scoped to the component classes used in the HTML.

(function(){
    // year
    const yearEl = document.getElementById('year');
    if (yearEl) {
        yearEl.textContent = new Date().getFullYear();
    }

    // navbar scroll toggle - use the nav-component class
    const nav = document.querySelector('.nav-component');
    function handleNav(){
        if(!nav) return;
        if(window.scrollY > 40) nav.classList.add('scrolled');
        else nav.classList.remove('scrolled');
    }
    handleNav();
    window.addEventListener('scroll', handleNav, { passive: true });

    // reveal about container when enters viewport
    const aboutContainer = document.querySelector('.about-container');
    if(aboutContainer && 'IntersectionObserver' in window){
        const obs = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if(entry.isIntersecting){
                    aboutContainer.classList.add('visible');
                    obs.disconnect();
                }
            });
        }, { threshold: 0.08 });
        obs.observe(aboutContainer);
    } else if (aboutContainer) {
        // Fallback: reveal immediately if observer not available
        aboutContainer.classList.add('visible');
    }

    // smooth scroll for anchor links (only internal page anchors)
    document.querySelectorAll('a[href^="#"]').forEach(a=>{
        a.addEventListener('click', function(e){
            // ensure this is an internal anchor on the same page
            const href = this.getAttribute('href');
            if (!href || href.length === 1) return;
            const target = document.querySelector(href);
            if(target){
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                // focus for accessibility after scroll
                setTimeout(()=> {
                    try {
                        target.setAttribute('tabindex','-1');
                        target.focus();
                    } catch(e){}
                }, 600);
            }
        });
    });

    // Accessible keyboard helpers can be added here if needed
})
();
