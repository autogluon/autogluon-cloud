let sidebar_scroll_element = document.querySelector(".sidebar-scroll");

let saved_top = sessionStorage.getItem("sidebar-scroll-top");
if (saved_top !== null) {
  sidebar_scroll_element.scrollTop = parseInt(saved_top, 10);
}

window.addEventListener("beforeunload", () => {
  sessionStorage.setItem("sidebar-scroll-top", sidebar_scroll_element.scrollTop);
});

// Landing page: open external links (AutoGluon, SageMaker, DLC, ...) in a new tab
document.querySelectorAll("section#autogluon-cloud a.reference.external").forEach((link) => {
  link.target = "_blank";
  link.rel = "noopener";
});
