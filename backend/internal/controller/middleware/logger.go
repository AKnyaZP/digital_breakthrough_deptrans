package middleware

import (
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func DefaultRequestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		// Process request
		c.Next()

		// After request
		latency := time.Since(start)
		statusCode := c.Writer.Status()
		var e *zerolog.Event

		if len(c.Errors) > 0 {
			e = log.Error().Err(c.Errors.Last())
		} else {
			e = log.Info()
		}

		e.
			Str("method", c.Request.Method).
			Str("host", c.Request.Host).
			Str("URI", c.Request.RequestURI).
			Int("status", statusCode).
			Str("latency", latency.String()).
			Msg("incoming request")
	}
}

func SetLogsFile() *os.File {
	file, err := os.OpenFile("logs/requests.log", os.O_APPEND|os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		panic(err)
	}
	return file
}
