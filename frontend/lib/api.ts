/**
 * Wrapper autenticado para chamadas à API.
 * Adiciona automaticamente o token JWT do localStorage e
 * redireciona para /login quando recebe 401.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export async function apiFetch(
  path: string,
  options: RequestInit = {},
): Promise<Response> {
  const token =
    typeof window !== 'undefined'
      ? localStorage.getItem('access_token')
      : null;

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((options.headers as Record<string, string>) || {}),
  };

  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: 'include',
  });

  // Token expirado ou inválido — limpar e redirecionar
  if (res.status === 401) {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    throw new Error('Unauthorized');
  }

  return res;
}

/**
 * Wrapper que já faz o parse JSON e lança erro se a resposta não for ok.
 */
export async function apiFetchJson<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await apiFetch(path, options);

  if (!res.ok) {
    const error = await res.json().catch(() => ({ message: 'Erro desconhecido' }));
    throw new Error(error.message || `HTTP ${res.status}`);
  }

  return res.json();
}
