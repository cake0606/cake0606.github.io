const storageKey = "site-theme";
const body = document.body;
const themeToggle = document.querySelector(".theme-toggle");
const themeToggleText = document.querySelector(".theme-toggle-text");
const header = document.querySelector(".site-header");
const navLinks = Array.from(document.querySelectorAll(".nav-link"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

function setTheme(theme) {
  if (theme === "light") {
    body.setAttribute("data-theme", "light");
    if (themeToggleText) {
      themeToggleText.textContent = "Light";
    }
  } else {
    body.removeAttribute("data-theme");
    if (themeToggleText) {
      themeToggleText.textContent = "Dark";
    }
  }

  localStorage.setItem(storageKey, theme);
}

function getPreferredTheme() {
  const storedTheme = localStorage.getItem(storageKey);
  if (storedTheme === "light" || storedTheme === "dark") {
    return storedTheme;
  }

  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function updateHeaderState() {
  header?.classList.toggle("is-scrolled", window.scrollY > 8);
}

function setActiveLink(id) {
  navLinks.forEach((link) => {
    link.classList.toggle("is-active", link.getAttribute("href") === `#${id}`);
  });
}

if (sections.length > 0) {
  const observer = new IntersectionObserver(
    (entries) => {
      const visibleEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (visibleEntry) {
        setActiveLink(visibleEntry.target.id);
      }
    },
    {
      rootMargin: "-28% 0px -55% 0px",
      threshold: [0.05, 0.25, 0.5]
    }
  );

  sections.forEach((section) => observer.observe(section));
}

themeToggle?.addEventListener("click", () => {
  const nextTheme = body.getAttribute("data-theme") === "light" ? "dark" : "light";
  setTheme(nextTheme);
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    setActiveLink(link.getAttribute("href").slice(1));
  });
});

window.addEventListener("scroll", updateHeaderState, { passive: true });

setTheme(getPreferredTheme());
updateHeaderState();
setActiveLink(window.location.hash.slice(1) || "projects");
