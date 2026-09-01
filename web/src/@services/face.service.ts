import { api } from "@services/api.instance";
import type { User, VerificationResult } from "@interfaces/face";

export const FaceService = {
  async getUser(userId: string, signal?: AbortSignal): Promise<User> {
    const response = await api.get<User>(`/users/${encodeURIComponent(userId)}`, { signal });
    return response.data;
  },

  profileImageUrl(userId: string): string {
    return `${api.defaults.baseURL}/users/${encodeURIComponent(userId)}/profile-image`;
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
