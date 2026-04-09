# MCP CRM Imobiliario (Postgres) - Documentacao Completa

> Referencia detalhada de todas as ferramentas do servidor MCP crm-imobiliario
> para gerenciamento do CRM imobiliario com banco de dados PostgreSQL (NeonDB).
>
> **Total: 35 ferramentas** organizadas em 10 categorias.

---

## Sumario

1. Banco de Dados - Estrutura
2. Banco de Dados - Performance e Manutencao
3. Clientes
4. Leads
5. Imoveis e Apartamentos
6. Usuarios e Corretores
7. Tarefas (Kanban)
8. Agendamentos
9. SLA e Notificacoes
10. Relatorios

---

## 1. Banco de Dados - Estrutura

### 1.1 query

Executa SQL arbitrario no banco de dados. Suporta SELECT, INSERT, UPDATE, DELETE e DDL.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| sql | string | **Sim** | SQL statement a executar |
| params | array | Nao | Parametros para queries parametrizadas (, ...) |
| timeout_ms | number | Nao | Timeout em milissegundos (padrao: 30000) |

**Exemplos:**


---

### 1.2 list_tables

Lista todas as tabelas do banco com contagem de linhas, tamanhos e tamanhos de indices.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| schema | string | Nao | Nome do schema (padrao: public) |

---

### 1.3 describe_table

Mostra schema completo de uma tabela: colunas, tipos, constraints, indices e foreign keys.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| table | string | **Sim** | Nome da tabela |
| schema | string | Nao | Nome do schema (padrao: public) |

---

### 1.4 list_enums

Lista todos os tipos enum customizados e seus valores.

**Parametros:** Nenhum.

---

### 1.5 list_relationships

Mostra todos os relacionamentos de foreign key entre tabelas.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| table | string | Nao | Filtrar por nome da tabela |

---

## 2. Banco de Dados - Performance e Manutencao

### 2.1 database_stats

Saude geral do banco: tamanho, cache hit rate, conexoes, commits/rollbacks, uptime.

**Parametros:** Nenhum.

---

### 2.2 table_stats

Estatisticas de manutencao: dead tuples, status do vacuum, bloat, contagem de scans.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| table | string | Nao | Filtrar por tabela (omitir = todas) |

---

### 2.3 index_usage

Analisa uso de indices: mais/menos usados, indices nao utilizados e tamanhos.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| table | string | Nao | Filtrar por tabela |

---

### 2.4 explain_query

Executa EXPLAIN ANALYZE em uma query SQL para ver o plano de execucao.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| sql | string | **Sim** | Query SQL a explicar |

---

### 2.5 running_queries

Mostra queries em execucao: duracao, estado e wait events.

**Parametros:** Nenhum.

---

### 2.6 kill_query

Termina uma query em execucao pelo PID (Process ID).

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| pid | number | **Sim** | Process ID da query a terminar |

---

### 2.7 vacuum_table

Executa VACUUM ANALYZE em uma tabela para recuperar dead rows e atualizar estatisticas.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| table | string | **Sim** | Nome da tabela |
| full | boolean | Nao | VACUUM FULL (bloqueia tabela, reescreve dados). Padrao: false |

---

## 3. Clientes

### 3.1 search_clients

Busca clientes por nome, email, telefone ou CPF (ILIKE). Suporta filtros por status, fonte, corretor e WhatsApp.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| search | string | Nao | Termo de busca (full_name, email, phone, cpf) |
| status | string | Nao | Filtrar por status do cliente |
| source | string | Nao | Filtrar por fonte do lead |
| broker_id | number | Nao | Filtrar por corretor (sistema_users.id) |
| has_whatsapp | boolean | Nao | Filtrar por disponibilidade de WhatsApp |
| limit | number | Nao | Max resultados (padrao: 20) |
| offset | number | Nao | Offset para paginacao |

---

### 3.2 get_client

Detalhes completos de um cliente: perfil, info do corretor, ultimas 10 notas, agendamentos, vendas, visitas e leads associados.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| client_id | number | **Sim** | ID do cliente |

---

### 3.3 client_stats

Estatisticas agregadas de clientes: contagem por status, por fonte e por corretor. Filtros opcionais por corretor e periodo.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| broker_id | number | Nao | Filtrar por corretor |
| period | string | Nao | Periodo: "30d", "90d", "1y" (baseado em created_at) |

---

### 3.4 client_timeline

Timeline unificada de todos os eventos de um cliente (notas, agendamentos, visitas, vendas) ordenada por data decrescente.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| client_id | number | **Sim** | ID do cliente |
| limit | number | Nao | Max eventos (padrao: 50) |

---

### 3.5 add_client_note

Adiciona uma nova nota/anotacao ao registro de um cliente.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| client_id | number | **Sim** | ID do cliente |
| user_id | number | **Sim** | ID do autor da nota (sistema_users.id) |
| text | string | **Sim** | Conteudo da nota |

---

## 4. Leads

### 4.1 search_leads

Busca leads por nome, email ou telefone (ILIKE). Filtros por status, fonte, corretor e score minimo.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| search | string | Nao | Termo de busca (full_name, email, phone) |
| status | string | Nao | Filtrar por status do lead |
| source | string | Nao | Filtrar por fonte |
| broker_id | number | Nao | Filtrar por corretor |
| min_score | number | Nao | Leads com score >= este valor |
| limit | number | Nao | Max resultados (padrao: 20) |
| offset | number | Nao | Offset para paginacao |

---

### 4.2 get_lead

Detalhes completos de um lead: perfil, info corretor, associacao com cliente, entradas SLA cascata ativas e ultimos 10 SLA logs.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| lead_id | number | **Sim** | ID do lead |

---

### 4.3 lead_pipeline

Visao geral do pipeline: contagem de leads agrupada por status. Filtros opcionais por corretor e fonte.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| broker_id | number | Nao | Filtrar por corretor |
| source | string | Nao | Filtrar por fonte |

---

### 4.4 lead_sources

Analise de fontes de leads: contagem e score medio agrupado por fonte. Filtro por periodo.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| date_from | string | Nao | Data inicio (ISO, baseado em created_at) |
| date_to | string | Nao | Data fim (ISO) |

---

## 5. Imoveis e Apartamentos

### 5.1 search_properties

Busca empreendimentos por nome, bairro ou cidade. Filtros por tipo, faixa de preco, cidade, bairro e zona. Retorna contagem de apartamentos.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| search | string | Nao | Busca por nome, bairro, cidade (ILIKE) |
| property_type | string | Nao | Filtrar por tipo_imovel |
| city | string | Nao | Filtrar por cidade_empreendimento |
| neighborhood | string | Nao | Filtrar por bairro_empreendimento |
| zone | string | Nao | Filtrar por zona_empreendimento |
| min_price | number | Nao | Preco minimo |
| max_price | number | Nao | Preco maximo |
| limit | number | Nao | Max resultados (padrao: 20) |
| offset | number | Nao | Offset para paginacao |

---

### 5.2 get_property

Detalhes completos de um empreendimento: todos os apartamentos, info da construtora e contatos da construtora.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| property_id | number | **Sim** | id_empreendimento |

---

### 5.3 property_availability

Lista apartamentos disponiveis filtrados por status. Filtros por empreendimento, quartos minimos e preco maximo.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| property_id | number | Nao | Filtrar por id_empreendimento |
| status | string | Nao | Status do apartamento (ex: "disponivel") |
| min_rooms | number | Nao | Minimo de quartos |
| max_price | number | Nao | Preco maximo (valor_venda_apartamento) |

---

### 5.4 property_price_analysis

Estatisticas de preco (min, max, media) agrupadas por bairro e zona. Filtros por bairro ou zona.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| neighborhood | string | Nao | Filtrar por bairro_empreendimento |
| zone | string | Nao | Filtrar por zona_empreendimento |

---

### 5.5 search_apartments

Busca direta de apartamentos com filtros por quartos, area, preco e status. Inclui info do empreendimento (localizacao).

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| rooms | number | Nao | Numero exato de quartos_apartamento |
| min_area | number | Nao | Area privativa minima |
| max_price | number | Nao | Preco maximo (valor_venda_apartamento) |
| status | string | Nao | Filtrar por status_apartamento |
| limit | number | Nao | Max resultados (padrao: 20) |
| offset | number | Nao | Offset para paginacao |

---

## 6. Usuarios e Corretores

### 6.1 list_users

Lista usuarios do sistema. Filtros por role, departamento e status ativo. Exclui password_hash.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| role | string | Nao | Filtrar por role |
| department | string | Nao | Filtrar por departamento |
| is_active | boolean | Nao | Filtrar por status ativo |

---

### 6.2 user_schedule

Horarios de trabalho de um usuario: dia da semana, hora inicio, hora fim, dia integral.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| user_id | number | **Sim** | ID do usuario |

---

### 6.3 broker_performance

Metricas de performance do corretor: total clientes, leads, vendas, valor total, taxa de conversao e agendamentos.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| broker_id | number | Nao | Filtrar por corretor |
| period | enum | Nao | Periodo: 30d (padrao), 90d, 1y |

---

## 7. Tarefas (Kanban)

### 7.1 get_board

Retorna board com suas listas e contagem de cards. Se nenhum board_id, retorna o primeiro ativo. Inclui stats: total cards, concluidos, atrasados.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| board_id | number | Nao | ID do board (padrao: primeiro ativo) |

---

### 7.2 list_tasks

Lista task cards com filtros por board, lista, responsavel, prioridade e arquivamento. Inclui nome da lista e usuario.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| board_id | number | Nao | Filtrar por board |
| list_id | number | Nao | Filtrar por lista |
| assigned_to | number | Nao | Filtrar por responsavel |
| priority | string | Nao | low, medium, high, urgent |
| is_archived | boolean | Nao | Incluir arquivados (padrao: false) |
| limit | number | Nao | Max resultados (padrao: 50) |
| offset | number | Nao | Offset paginacao |

---

### 7.3 create_task

Cria novo card de tarefa em uma lista. Retorna o card criado.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| list_id | number | **Sim** | ID da lista |
| title | string | **Sim** | Titulo do card |
| description | string | Nao | Descricao |
| assigned_to | number | Nao | ID do responsavel |
| priority | enum | Nao | low, medium (padrao), high, urgent |
| due_date | string | Nao | Data limite (ISO 8601) |
| estimated_hours | number | Nao | Horas estimadas |
| tags | array | Nao | Array de tags |

---

### 7.4 update_task

Atualiza card. Apenas campos fornecidos sao atualizados. Retorna card atualizado.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| card_id | number | **Sim** | ID do card |
| title | string | Nao | Novo titulo |
| description | string | Nao | Nova descricao |
| list_id | number | Nao | Mover para outra lista |
| assigned_to | number | Nao | Reatribuir |
| priority | enum | Nao | low, medium, high, urgent |
| due_date | string | Nao | Nova data limite |
| estimated_hours | number | Nao | Horas estimadas |
| actual_hours | number | Nao | Horas reais |
| tags | array | Nao | Substituir tags |
| is_archived | boolean | Nao | Arquivar/desarquivar |
| completed_at | string | Nao | Data conclusao (ISO), ou vazio para limpar |

---

## 8. Agendamentos

### 8.1 list_appointments

Lista agendamentos com filtros flexiveis: cliente, corretor, status, tipo, somente futuros, intervalo de datas. Inclui nomes do cliente e usuario.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| client_id | number | Nao | Filtrar por cliente |
| broker_id | number | Nao | Filtrar por corretor |
| status | string | Nao | Filtrar por status |
| type | string | Nao | Filtrar por tipo de agendamento |
| upcoming_only | boolean | Nao | Somente futuros (padrao: false) |
| date_from | string | Nao | Data inicio (ISO) |
| date_to | string | Nao | Data fim (ISO) |
| limit | number | Nao | Max resultados (padrao: 50) |

---

## 9. SLA e Notificacoes

### 9.1 sla_status

Mostra entradas SLA cascata ativas (ativo=true) com tempo restante ate o deadline. Filtros por corretor e horas ate expiracao.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| broker_id | number | Nao | Filtrar por corretor (usuario_id) |
| expiring_within_hours | number | Nao | Somente SLAs expirando dentro de X horas |
| limit | number | Nao | Max resultados (padrao: 50) |

---

### 9.2 sla_expiring

Lista SLAs expirando dentro de X horas (padrao: 4). Inclui nomes do usuario e cliente.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| hours | number | Nao | Horas ate expiracao (padrao: 4) |
| broker_id | number | Nao | Filtrar por corretor |

---

### 9.3 notifications

Notificacoes de um usuario. Filtro por nao-lidas e limite.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| user_id | number | **Sim** | ID do usuario |
| unread_only | boolean | Nao | Somente nao-lidas (padrao: false) |
| limit | number | Nao | Max resultados (padrao: 20) |

---

### 9.4 whatsapp_status

Status de todas as instancias WhatsApp: nome, status, usuario associado e ultima conexao.

**Parametros:** Nenhum.

---

## 10. Relatorios

### 10.1 daily_report

Resumo diario: novos leads, novos clientes, valor total de vendas, agendamentos e expiracoes SLA. Filtros por data e corretor.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| date | string | Nao | Data do relatorio (YYYY-MM-DD, padrao: hoje) |
| broker_id | number | Nao | Filtrar por corretor |

---

### 10.2 sales_report

Relatorio de vendas: valor total, comissao e total_commission agrupado por corretor. Filtros por corretor e periodo.

| Parametro | Tipo | Obrigatorio | Descricao |
|-----------|------|:-----------:|-----------|
| broker_id | number | Nao | Filtrar por corretor |
| date_from | string | Nao | Data inicio (ISO, baseado em sold_at) |
| date_to | string | Nao | Data fim (ISO) |

---

## Referencia Rapida

### Resumo por Categoria

| Categoria | Qtd | Ferramentas |
|-----------|-----|-------------|
| DB Estrutura | 5 | query, list_tables, describe_table, list_enums, list_relationships |
| DB Performance | 7 | database_stats, table_stats, index_usage, explain_query, running_queries, kill_query, vacuum_table |
| Clientes | 5 | search_clients, get_client, client_stats, client_timeline, add_client_note |
| Leads | 4 | search_leads, get_lead, lead_pipeline, lead_sources |
| Imoveis | 5 | search_properties, get_property, property_availability, property_price_analysis, search_apartments |
| Usuarios | 3 | list_users, user_schedule, broker_performance |
| Tarefas | 3 | get_board, list_tasks, create_task, update_task |
| Agendamentos | 1 | list_appointments |
| SLA/Notificacoes | 4 | sla_status, sla_expiring, notifications, whatsapp_status |
| Relatorios | 2 | daily_report, sales_report |

### Fluxo Tipico de Uso

1. **Explorar banco:** list_tables -> describe_table -> list_enums
2. **Buscar cliente:** search_clients -> get_client -> client_timeline
3. **Pipeline de leads:** lead_pipeline -> search_leads -> get_lead
4. **Buscar imovel:** search_properties -> property_availability -> search_apartments
5. **Performance:** broker_performance -> sales_report -> daily_report
6. **Monitorar SLA:** sla_expiring -> sla_status
7. **Saude do banco:** database_stats -> table_stats -> index_usage -> running_queries

### Periodos Suportados

| Valor | Descricao |
|-------|-----------|
| 30d | Ultimos 30 dias |
| 90d | Ultimos 90 dias |
| 1y | Ultimo ano |

### Prioridades de Tarefas

| Valor | Descricao |
|-------|-----------|
| low | Baixa |
| medium | Media (padrao) |
| high | Alta |
| urgent | Urgente |

---
> Documento gerado em 09/04/2026 com base nos schemas do MCP crm-imobiliario.
