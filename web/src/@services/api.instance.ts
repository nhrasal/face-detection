import axios, { AxiosError } from "axios";
import type { ApiErrorBody } from "@interfaces/face";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code = "REQUEST_FAILED",
  ) {
    super(message);
  }
}

export const api = axios.create({
  baseURL: (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, ""),
  timeout: 30_000,
  headers: { Accept: "application/json" },
});

api.interceptors.response.use(
  (response) => response,
  (caught: AxiosError<ApiErrorBody>) => {
    if (caught.code === "ERR_CANCELED") return Promise.reject(caught);
    const body = caught.response?.data;
    const validationMessage = Array.isArray(body?.detail) ? body.detail[0]?.msg : body?.detail;
    return Promise.reject(
      new ApiError(
        body?.error?.message || validationMessage || "The service could not complete the request.",
        caught.response?.status || 0,
        body?.error?.code,
      ),
    );
  },
);

/** True when a request was aborted by us, not by the network or the service. */
export const isCanceled = (caught: unknown): boolean =>
  caught instanceof AxiosError && caught.code === "ERR_CANCELED";
