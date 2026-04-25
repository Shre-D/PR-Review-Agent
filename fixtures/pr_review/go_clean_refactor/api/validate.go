package api

import "errors"

func ValidateID(id string) error {
    if id == "" {
        return errors.New("missing id")
    }
    return nil
}
