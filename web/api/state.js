"use strict";

const { createHandlers, defaultRedis } = require("../lib/handlers");

module.exports = createHandlers({ getRedis: () => defaultRedis() }).state;
