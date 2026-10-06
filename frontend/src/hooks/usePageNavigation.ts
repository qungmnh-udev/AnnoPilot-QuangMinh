import { useCallback, useEffect, useState } from "react";

export const pagePaths = {
  Home: "/",
  Dashboard: "/dashboard",
  Dataset: "/dataset",
  Difficulty: "/difficulty",
  "Smart Sampling": "/smart-sampling",
  Workload: "/workload",
  "Review Queue": "/review-queue",
  "Model QC": "/model-qc",
  Settings: "/settings",
} as const;
export type Page = keyof typeof pagePaths;

export function isDatasetPage(page: Page) {
  return page !== "Home" && page !== "Settings";
}

function currentPage(): Page {
  const path = window.location.pathname.replace(/\/$/, "") || "/";
  return (
    (Object.keys(pagePaths) as Page[]).find(
      (page) => pagePaths[page] === path,
    ) || "Home"
  );
}

// Keep the existing page components/state; no routing dependency is needed.
export function usePageNavigation() {
  const [page, setPage] = useState<Page>(currentPage);
  useEffect(() => {
    const onPopState = () => setPage(currentPage());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);
  const navigate = useCallback((next: Page, replace = false) => {
    const path = pagePaths[next];
    if (window.location.pathname !== path) {
      window.history[replace ? "replaceState" : "pushState"]({}, "", path);
    }
    setPage(next);
  }, []);
  return { page, navigate };
}
