package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"time"

	models "gemini-web-to-api/internal/commons/models"
	utils "gemini-web-to-api/internal/commons/utils"
	"gemini-web-to-api/internal/modules/openai/dto"
	"gemini-web-to-api/internal/modules/providers"

	"github.com/gofiber/fiber/v3"
	"go.uber.org/zap"
)

type OpenAIController struct {
	service *OpenAIService
	log     *zap.Logger
}

func NewOpenAIController(service *OpenAIService) *OpenAIController {
	return &OpenAIController{
		service: service,
		log:     zap.NewNop(),
	}
}

// SetLogger sets the logger for this handler
func (h *OpenAIController) SetLogger(log *zap.Logger) {
	h.log = log
}

// GetModelData returns raw model data for internal use (e.g. unified list)
func (h *OpenAIController) GetModelData() []models.ModelData {
	availableModels := h.service.ListModels()

	var data []models.ModelData
	for _, m := range availableModels {
		data = append(data, models.ModelData{
			ID:      m.ID,
			Object:  "model",
			Created: m.Created,
			OwnedBy: m.OwnedBy,
		})
	}
	return data
}

// HandleModels returns the list of supported models
// @Summary List OpenAI Models
// @Description Returns a list of models supported by the OpenAI-compatible API
// @Tags OpenAI
// @Accept json
// @Produce json
// @Success 200 {object} models.ModelListResponse
// @Router /openai/v1/models [get]
func (h *OpenAIController) HandleModels(c fiber.Ctx) error {
	data := h.GetModelData()

	return c.JSON(models.ModelListResponse{
		Object: "list",
		Data:   data,
	})
}

// HandleChatCompletions accepts requests in OpenAI format
// @Summary Chat Completions (OpenAI)
// @Description Generates a completion for the chat message. Supports both standard JSON and streaming (SSE) response.
// @Tags OpenAI
// @Accept json
// @Produce json
// @Produce text/event-stream
// @Param request body dto.ChatCompletionRequest true "Chat Completion Request"
// @Success 200 {object} dto.ChatCompletionResponse
// @Success 200 {string} string "SSE stream of dto.ChatCompletionChunk JSON objects"
// @Failure 400 {object} map[string]interface{}
// @Failure 500 {object} map[string]interface{}
// @Router /openai/v1/chat/completions [post]
func (h *OpenAIController) HandleChatCompletions(c fiber.Ctx) error {
	var req dto.ChatCompletionRequest
	if err := c.Bind().Body(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(utils.ErrorToResponse(fmt.Errorf("invalid request body: %w", err), "invalid_request_error"))
	}

	if req.Stream {
		c.Set("Content-Type", "text/event-stream")
		c.Set("Cache-Control", "no-cache")
		c.Set("Connection", "keep-alive")
		c.Set("X-Accel-Buffering", "no")

		c.RequestCtx().SetBodyStreamWriter(func(w *bufio.Writer) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()

			err := h.service.CreateChatCompletionStream(ctx, req, func(chunk dto.ChatCompletionChunk) bool {
				return utils.SendSSEEvent(w, h.log, chunk)
			})
			if err != nil {
				h.log.Error("CreateChatCompletionStream failed", zap.Error(err), zap.String("model", req.Model))
				// Optionally send actual OpenAI error JSON here, but for now just close.
				return
			}
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		})

		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	response, err := h.service.CreateChatCompletion(ctx, req)
	if err != nil {
		h.log.Error("CreateChatCompletion failed", zap.Error(err), zap.String("model", req.Model))
		return c.Status(fiber.StatusInternalServerError).JSON(utils.ErrorToResponse(err, "api_error"))
	}

	if req.Stream {
		return h.streamResponse(c, response)
	}

	return c.JSON(response)
}

func (h *OpenAIController) streamResponse(c fiber.Ctx, response *dto.ChatCompletionResponse) error {
	c.Set(fiber.HeaderContentType, "text/event-stream")
	c.Set(fiber.HeaderCacheControl, "no-cache")
	c.Set("Connection", "keep-alive")
	c.Set("X-Accel-Buffering", "no")

	text := ""
	if len(response.Choices) > 0 {
		text = response.Choices[0].Message.Content
	}

	return c.SendStreamWriter(func(w *bufio.Writer) {
		writeChunk := func(chunk dto.ChatCompletionChunk) {
			data, err := json.Marshal(chunk)
			if err != nil {
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			w.Flush()
		}

		// First chunk: establish role
		writeChunk(dto.ChatCompletionChunk{
			ID:      response.ID,
			Object:  "chat.completion.chunk",
			Created: response.Created,
			Model:   response.Model,
			Choices: []dto.ChunkChoice{
				{Index: 0, Delta: models.Delta{Role: "assistant"}},
			},
		})

		// Content chunk
		writeChunk(dto.ChatCompletionChunk{
			ID:      response.ID,
			Object:  "chat.completion.chunk",
			Created: response.Created,
			Model:   response.Model,
			Choices: []dto.ChunkChoice{
				{Index: 0, Delta: models.Delta{Content: text}},
			},
		})

		// Final chunk with finish_reason
		writeChunk(dto.ChatCompletionChunk{
			ID:      response.ID,
			Object:  "chat.completion.chunk",
			Created: response.Created,
			Model:   response.Model,
			Choices: []dto.ChunkChoice{
				{Index: 0, Delta: models.Delta{}, FinishReason: "stop"},
			},
		})

		fmt.Fprint(w, "data: [DONE]\n\n")
		w.Flush()
	})
}

func (h *OpenAIController) convertToOpenAIFormat(response *providers.Response, model string) dto.ChatCompletionResponse {
	return dto.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []dto.Choice{
			{
				Index: 0,
				Message: models.Message{
					Role:    "assistant",
					Content: response.Text,
				},
				FinishReason: "stop",
			},
		},
		Usage: models.Usage{
			PromptTokens:     0,
			CompletionTokens: 0,
			TotalTokens:      0,
		},
	}
}

// Register registers the OpenAI routes onto the provided group
func (c *OpenAIController) Register(group fiber.Router) {
	group.Get("/models", c.HandleModels)
	group.Post("/chat/completions", c.HandleChatCompletions)
}
