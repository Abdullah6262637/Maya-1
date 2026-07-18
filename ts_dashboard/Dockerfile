# Production Dockerfile for TS Dashboard Server
FROM node:20-alpine

WORKDIR /app

# Copy package config files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies (clean install)
RUN npm ci

# Copy source code and assets
COPY server.ts ./
COPY public/ ./public/

# Compile TypeScript to JavaScript (dist/server.js)
RUN npm run build

# Expose Express and WebSocket port
EXPOSE 4001

# Run compiled Node application
CMD ["node", "dist/server.js"]
