import React from "react"
import ReactDOM from "react-dom"
import CustomSlider from "./CustomSlider"

// Lots of import to define a Styletron engine and load the light theme of baseui
import { Provider as StyletronProvider, DebugEngine } from "styletron-react";
import { Client as Styletron } from "styletron-engine-atomic";
import { ThemeProvider, LightTheme } from "baseui";

const debug = process.env.NODE_ENV === "production" ? void 0 : new DebugEngine();

// 1. Create a client engine instance
const engine = new Styletron();

// 2. Provide the engine to the app
// debug engine needs inlined source maps

ReactDOM.render(
  <React.StrictMode>
    <StyletronProvider value={engine} debug={debug} debugAfterHydration>
      <ThemeProvider theme={LightTheme}>
        <CustomSlider />
      </ThemeProvider>
    </StyletronProvider>
  </React.StrictMode>
  ,
  document.getElementById("root")
)