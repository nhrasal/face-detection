import { api } from "@services/api.instance";
import type { DetectResult, User, UserSearchResponse, VerificationResult } from "@interfaces/face";

export const FaceService = {
  async getUser(userId: string, signal?: AbortSignal): Promise<User> {
    const response = await api.get<User>(`/users/${encodeURIComponent(userId)}`, { signal });
    return response.data;
  },

  async searchUsers(query: string, signal?: AbortSignal): Promise<User[]> {
    const response = await api.get<UserSearchResponse>("/users", {
      params: { search: query },
      signal,
    });
    return response.data.items;
  },

  /**
   * Detect on one live preview frame.
   *
   * A short timeout on purpose: a frame that has not come back by the time the
   * next one is due is already stale, and waiting 30s for it would stall the
   * preview loop behind a request whose answer no longer describes the scene.
   */
  async detectFrame(frame: Blob, signal?: AbortSignal): Promise<DetectResult> {
    const form = new FormData();
    form.append("frame", frame, "frame.jpg");
    const response = await api.post<DetectResult>("/face/detect/frame", form, {
      signal,
      timeout: 5_000,
    });
    return response.data;
  },

  profileImageUrl(userId: string): string {
    return `${api.defaults.baseURL}/users/${encodeURIComponent(userId)}/profile-image`;
  },

  async createUser(
    externalId: string,
    name: string,
    profileImage: File,
    signal?: AbortSignal,
  ): Promise<User> {
    const form = new FormData();
    form.append("external_id", externalId);
    form.append("name", name);
    form.append("profile_image", profileImage);
    const response = await api.post<User>("/users", form, { signal });
    return response.data;
  },

  async verifyUser(
    userId: string,
    candidate: File,
    signal?: AbortSignal,
  ): Promise<VerificationResult> {
    const form = new FormData();
    form.append("candidate_image", candidate);
    const response = await api.post<VerificationResult>(
      `/users/${encodeURIComponent(userId)}/verify`,
      form,
      { signal },
    );
    return response.data;
  },
};
