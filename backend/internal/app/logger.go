package app

import (
	"os"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func SetLogger(lev string) {
	level, err := zerolog.ParseLevel(lev)
	if err != nil {
		level = zerolog.DebugLevel
	}

	log.Logger = zerolog.New(
		zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339},
	).
		Level(level).
		With().
		Timestamp().
		//		Caller().
		Logger()
}
