# MCP MinIO - Documentacao Completa de Ferramentas

> Referencia detalhada de todas as ferramentas do servidor MCP minio
> para gerenciamento de armazenamento de objetos (Object Storage) compativel com S3.
>
> **Total: 30 ferramentas** organizadas em 8 categorias.

---

## Sumario

1. Servidor
2. Buckets - Gerenciamento
3. Buckets - Configuracao
4. Objetos - Leitura
5. Objetos - Escrita
6. Objetos - Tags
7. URLs e Acesso
8. Uploads Incompletos

---

## 1. Servidor

### 1.1 minio_server_info

Retorna informacoes de configuracao do servidor MinIO e status de conectividade.

**Parametros:** Nenhum.

---

## 2. Buckets - Gerenciamento

### 2.1 minio_list_buckets

Lista todos os buckets disponiveis no servidor MinIO com data de criacao.

**Parametros:** Nenhum.

---

### 2.2 minio_create_bucket

Cria um novo bucket no MinIO.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket (3-63 caracteres, lowercase) |
| region | string | Nao | Regiao (padrao: us-east-1) |

---

### 2.3 minio_delete_bucket

Remove um bucket vazio. O bucket **deve estar vazio** antes da exclusao.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket a remover |

---

### 2.4 minio_bucket_exists

Verifica se um bucket existe. Retorna true/false.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket |

---

### 2.5 minio_bucket_summary

Estatisticas de um bucket: total de objetos, tamanho acumulado, breakdown por extensao. Pode ser lento para buckets grandes.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | Nao | Nome do bucket (padrao: configurado) |
| prefix | string | Nao | Prefixo/pasta para limitar analise |
| max_objects | integer | Nao | Max objetos a analisar (padrao: 5000, max: 50000) |

---

## 3. Buckets - Configuracao

### 3.1 minio_get_bucket_policy

Retorna a politica de acesso IAM (JSON) de um bucket.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket |

---

### 3.2 minio_set_bucket_policy

Define a politica de acesso IAM de um bucket. Use policy_json como string JSON valida.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket |
| policy_json | string | **Sim** | Politica IAM em formato JSON string |

---

### 3.3 minio_get_bucket_tags

Retorna as tags (metadados chave-valor) de um bucket.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket |

---

### 3.4 minio_set_bucket_tags

Define tags em um bucket. **Substitui todas as tags existentes.**

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket |
| tags | object | **Sim** | Pares chave-valor. Ex: {env:prod, team:devops} |

---

### 3.5 minio_get_bucket_versioning

Retorna a configuracao de versionamento de um bucket.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket |

---

### 3.6 minio_set_bucket_versioning

Ativa ou suspende o versionamento de objetos em um bucket.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | **Sim** | Nome do bucket |
| status | enum | **Sim** | Enabled = ativar, Suspended = suspender |

---

## 4. Objetos - Leitura

### 4.1 minio_list_objects

Lista objetos em um bucket com suporte a prefixo, busca recursiva e paginacao. Retorna nome, tamanho, etag e data de modificacao.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | Nao | Nome do bucket (padrao: configurado) |
| prefix | string | Nao | Prefixo/pasta. Ex: uploads/2024/ |
| recursive | boolean | Nao | true = todos subdiretorios, false = nivel atual (padrao) |
| max_keys | integer | Nao | Max objetos (padrao: 200, max: 10000) |

**Exemplo:** 

---

### 4.2 minio_get_object_info

Retorna metadados: tamanho, etag, data de modificacao, content-type e metadados customizados.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome do objeto. Ex: uploads/foto.jpg |
| bucket | string | Nao | Nome do bucket |

---

### 4.3 minio_get_object_json

Download e parse de conteudo JSON. Retorna objeto JSON parseado. **Limite: 5 MB.**

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho do arquivo .json |
| bucket | string | Nao | Nome do bucket |

**Exemplo:** 

---

### 4.4 minio_get_object_text

Download como texto (UTF-8). Para .txt, .csv, .html, .md, .xml, .yml, .log. **Limite: 5 MB.**

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome do objeto |
| bucket | string | Nao | Nome do bucket |
| encoding | enum | Nao | utf-8 (padrao), latin1, ascii |

**Exemplo:** 

---

### 4.5 minio_search_objects

Busca objetos por padrao de nome (substring, sufixo, extensao). Ex: .pdf, relatorio_, 2024.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| pattern | string | **Sim** | Padrao de busca (substring). Ex: .pdf, invoice_ |
| bucket | string | Nao | Nome do bucket |
| prefix | string | Nao | Prefixo base. Ex: uploads/ |
| max_results | integer | Nao | Max resultados (padrao: 100, max: 1000) |
| max_scan | integer | Nao | Max objetos a escanear (padrao: 5000, max: 50000) |
| case_sensitive | boolean | Nao | Diferenciar maiusculas (padrao: false) |

**Exemplo:** 

---

## 5. Objetos - Escrita

### 5.1 minio_put_object_json

Upload de objeto JSON com formatacao adequada.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome. Ex: data/config.json |
| data | object | **Sim** | Objeto JSON a serializar e fazer upload |
| bucket | string | Nao | Nome do bucket |
| pretty | boolean | Nao | true = formatado (padrao), false = minificado |
| metadata | object | Nao | Metadados customizados |

---

### 5.2 minio_put_object_text

Upload de conteudo texto. Para .txt, .csv, .md, .xml, .html.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome. Ex: reports/2024/relatorio.txt |
| content | string | **Sim** | Conteudo de texto |
| bucket | string | Nao | Nome do bucket |
| content_type | string | Nao | MIME type (padrao: text/plain; charset=utf-8) |
| metadata | object | Nao | Metadados customizados |

---

### 5.3 minio_copy_object

Copia objeto de origem para destino (mesmo servidor).

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| source_bucket | string | **Sim** | Bucket de origem |
| source_object | string | **Sim** | Caminho do objeto origem |
| dest_bucket | string | **Sim** | Bucket de destino |
| dest_object | string | **Sim** | Caminho do objeto destino |

---

### 5.4 minio_move_object

Move objeto (copy + delete). Operacao atomica.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| source_bucket | string | **Sim** | Bucket de origem |
| source_object | string | **Sim** | Caminho objeto origem |
| dest_bucket | string | **Sim** | Bucket de destino |
| dest_object | string | **Sim** | Caminho objeto destino |

---

### 5.5 minio_delete_object

Remove um unico objeto.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome do objeto |
| bucket | string | Nao | Nome do bucket |
| version_id | string | Nao | ID da versao (buckets com versionamento) |

---

### 5.6 minio_delete_objects

Remove multiplos objetos em uma operacao (ate 1000).

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| objects | array | **Sim** | Lista de caminhos (min: 1, max: 1000) |
| bucket | string | Nao | Nome do bucket |

---

## 6. Objetos - Tags

### 6.1 minio_get_object_tags

Retorna as tags (metadados chave-valor) de um objeto.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome do objeto |
| bucket | string | Nao | Nome do bucket |

---

### 6.2 minio_set_object_tags

Define ou substitui tags de um objeto.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome do objeto |
| tags | object | **Sim** | Pares chave-valor. Ex: {project:crm, type:avatar} |
| bucket | string | Nao | Nome do bucket |

---

### 6.3 minio_remove_object_tags

Remove todas as tags de um objeto.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome do objeto |
| bucket | string | Nao | Nome do bucket |
| version_id | string | Nao | ID da versao (versionamento) |

---

## 7. URLs e Acesso

### 7.1 minio_get_presigned_url

Gera URL pre-assinada para acesso temporario sem autenticacao. GET = download, PUT = upload.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome do objeto |
| bucket | string | Nao | Nome do bucket |
| method | enum | Nao | GET = download (padrao), PUT = upload |
| expiry_seconds | integer | Nao | Validade em segundos (padrao: 3600=1h, max: 604800=7dias) |

**Casos de uso:**
- GET: Compartilhar link temporario de download
- PUT: Permitir upload direto do cliente/frontend sem credenciais

---

### 7.2 minio_generate_public_url

Gera URL publica permanente (sem expiracao). O bucket **deve ter politica publica de leitura.**

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Caminho/nome do objeto |
| bucket | string | Nao | Nome do bucket |

---

## 8. Uploads Incompletos

### 8.1 minio_list_incomplete_uploads

Lista uploads multipart incompletos (abandonados) que ocupam espaco.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| bucket | string | Nao | Nome do bucket |
| prefix | string | Nao | Prefixo para filtrar |
| max_results | integer | Nao | Max resultados (padrao: 100, max: 1000) |

---

### 8.2 minio_abort_incomplete_upload

Aborta e remove um upload multipart incompleto. Use minio_list_incomplete_uploads primeiro.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| object | string | **Sim** | Chave (key) do upload incompleto |
| bucket | string | Nao | Nome do bucket |

---

## Referencia Rapida

### Estrutura MinIO


### Resumo por Categoria

| Categoria | Qtd | Ferramentas |
|-----------|-----|-------------|
| Servidor | 1 | server_info |
| Buckets Gerenciamento | 5 | list, create, delete, exists, summary |
| Buckets Configuracao | 6 | get/set policy, get/set tags, get/set versioning |
| Objetos Leitura | 5 | list, get_info, get_json, get_text, search |
| Objetos Escrita | 6 | put_json, put_text, copy, move, delete, delete_objects |
| Objetos Tags | 3 | get_tags, set_tags, remove_tags |
| URLs e Acesso | 2 | presigned_url, public_url |
| Uploads Incompletos | 2 | list_incomplete, abort_incomplete |

### Limites Importantes

| Limite | Valor |
|--------|-------|
| Nome de bucket | 3-63 caracteres, lowercase |
| Download texto/JSON | Max 5 MB |
| Delete em batch | Max 1000 objetos |
| URL pre-assinada | Max 7 dias (604800s) |
| Busca de objetos | Max 50000 escaneados |
| Bucket summary | Max 50000 analisados |

---
> Documento gerado em 09/04/2026 com base nos schemas do MCP minio.
