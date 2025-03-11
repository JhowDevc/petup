# PetUp - Sistema de Gerenciamento e Busca de Pets

Bem-vindo ao **PetFinder**, um sistema composto por três APIs integradas para autenticação de usuários, gerenciamento de informações de pets e busca avançada de pets perdidos por similaridade de imagem. Este projeto utiliza tecnologias modernas em Java (Spring Boot) e Python (Flask), combinando autenticação segura, persistência de dados e visão computacional.

## Visão Geral

O PetFinder é dividido em três componentes principais:
1. **User API**: Uma API REST em Java para autenticação e gerenciamento de usuários.
2. **Pet API**: Uma API REST em Java para cadastro, busca e exclusão de pets.
3. **Pet Search API**: Uma API em Python para busca de pets perdidos por similaridade de imagem usando visão computacional.

Esses componentes podem ser integrados para criar uma solução completa de rastreamento e recuperação de pets perdidos.

---

## User API

### Descrição
API REST para autenticação e gerenciamento de usuários, incluindo login tradicional, registro, recuperação de senha e autenticação via OAuth2 (Google).

### Tecnologias Utilizadas
- **Spring Boot**: Framework para construção da API.
- **Spring Security**: Autenticação e autorização com suporte a OAuth2 e JWT.
- **JavaMailSender**: Envio de e-mails para recuperação de senha (SMTP do Gmail).
- **JPA/Hibernate**: Persistência de dados em banco relacional (MySQL/PostgreSQL).
- **JWT (io.jsonwebtoken)**: Geração e validação de tokens de autenticação.
- **BCrypt**: Criptografia de senhas.
- **SLF4J**: Logging de eventos e erros.

### Estrutura Principal
- **EmailConfig**: Configuração do JavaMailSender.
- **SecurityConfig**: Configuração de segurança com filtros JWT.
- **AuthController**: Endpoints REST para autenticação.
- **JwtService**: Lógica de criação e validação de tokens.
- **JwtAuthenticationFilter**: Validação de tokens em requisições.
- **UserRepository**: Operações no banco de dados.

### Endpoints
- `POST /login`: Autentica um usuário e retorna um token JWT.
- `POST /register`: Registra um novo usuário.
- `POST /forgot-password`: Envia e-mail para recuperação de senha.
- `GET /oauth2/google`: Login via Google OAuth2.

### Propósito
Gerenciar usuários com segurança, fornecendo tokens JWT para acesso autenticado às demais APIs.

---

## Pet API

### Descrição
API REST para gerenciamento de informações de pets, incluindo cadastro, busca e exclusão.

### Tecnologias Utilizadas
- **Spring Boot**: Framework para a API.
- **Spring Web**: Endpoints REST e manipulação de requisições HTTP.
- **JPA/Hibernate**: Persistência da entidade `Pet`.
- **Lombok**: Redução de código boilerplate com anotações como `@Data`.
- **CORS**: Configurado para permitir origens específicas (ex.: `localhost:3000`).

### Estrutura Principal
- **CorsConfig**: Configuração global de CORS.
- **PetController**: Endpoints REST para gerenciamento de pets.
- **PetDTO**: Objeto de transferência de dados.
- **Pet**: Entidade JPA mapeada para o banco.

### Endpoints
- `GET /pets`: Busca pets por filtros ou imagem.
- `POST /pets`: Cadastra um novo pet com foto.
- `DELETE /pets/{id}`: Exclui um pet específico.

### Propósito
Permitir o gerenciamento de dados de pets de forma estruturada e segura, integrando-se com a autenticação da User API.

---

## Pet Search API

### Descrição
API Flask para busca de pets perdidos por similaridade de imagem, utilizando modelos de visão computacional.

### Tecnologias Utilizadas
- **Flask**: Framework leve para API REST.
- **MySQL Connector**: Conexão com banco MySQL.
- **OpenCLIP**: Extração de características visuais com modelo CLIP (ViT-L-14).
- **YOLOv8 (Ultralytics)**: Segmentação de animais em imagens.
- **rembg**: Remoção de fundo como fallback.
- **Pillow (PIL)**: Manipulação de imagens.
- **OpenCV (cv2)**: Pós-processamento de máscaras.
- **NumPy**: Cálculo de similaridade cosseno.
- **PyTorch**: Backend para execução do modelo CLIP.
- **Logging**: Registro de eventos.

### Estrutura Principal
- **Config**: Configurações de banco, modelo CLIP e limiar de similaridade (0.88).
- **Modelos**:
  - `model_clip`: CLIP para extração de características.
  - `model_yolo`: YOLOv8 para segmentação.
- **Funções Auxiliares**:
  - `segment_animal`: Segmenta o animal na imagem.
  - `extract_features`: Extrai características visuais.
  - `calculate_similarity`: Calcula similaridade entre vetores.
  - `detect_species`: Identifica a espécie do animal.
- **Endpoint**:
  - `POST /search`: Recebe uma imagem e retorna pets similares.

### Fluxo de Processamento
1. **Entrada**: Recebe uma imagem via POST.
2. **Detecção**: YOLOv8 identifica a espécie (ex.: cachorro, gato).
3. **Segmentação**: YOLOv8 ou rembg isola o animal da imagem.
4. **Extração**: OpenCLIP gera um vetor de características.
5. **Consulta**: Busca pets da mesma espécie no banco MySQL.
6. **Comparação**: Calcula similaridade cosseno entre vetores.
7. **Saída**: Retorna IDs de pets similares em JSON.

### Propósito
Facilitar a busca de pets perdidos comparando imagens enviadas com as armazenadas, utilizando visão computacional avançada.

---

## Integração dos Componentes

### Fluxo Geral
1. **User API**: Autentica usuários e gera tokens JWT.
2. **Pet API**: Usa os tokens JWT para proteger endpoints de cadastro e exclusão de pets.
3. **Pet Search API**: Pode ser chamada pela Pet API para realizar buscas por imagem, compartilhando o mesmo banco MySQL.

### Banco de Dados
- **MySQL**: Utilizado por todas as APIs para persistência de dados (usuários e pets).

### Possível Arquitetura
- Frontend (ex.: React) faz chamadas autenticadas para a `Pet API`.
- `Pet API` consulta a `Pet Search API` para buscas por imagem.
- Tokens JWT garantem segurança entre APIs e frontend.

---

## Pré-requisitos

### User API e Pet API (Java)
- Java 17+
- Maven
- MySQL/PostgreSQL
- Conta Gmail para envio de e-mails (SMTP)

### Pet Search API (Python)
- Python 3.8+
- MySQL
- Dependências:
  ```bash
  pip install flask mysql-connector-python open-clip-torch ultralytics rembg pillow opencv-python numpy torch
