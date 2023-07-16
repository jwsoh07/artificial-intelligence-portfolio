import React, { useState, useEffect } from "react";
import { Slider } from "baseui/slider";
import {
  Streamlit, withStreamlitConnection, ComponentProps
} from "streamlit-component-lib";

/**
 * Called by <CustomSlider />, renders the return value on screen.
 *
 * (props) => {code} is an arrow function, a shorter syntax for JS functions
 * equivalent to : function (props) { code; return <h1>Hello world</h1>};
 * or in Python, lambda props : return <h1>Hello world</h1>.
 *
 * When called, it will run then render on the browser the returned JSX block
 */
const CustomSlider = (props: ComponentProps) => {
  /**
 * Destructuring JSON objects is a good habit.
 */
  
  const [value, setValue] = useState([120]);

  useEffect(() => Streamlit.setFrameHeight());

  return (
    <>
      <Slider
        value={value}
        onChange={({ value }) => value && setValue(value)}
        onFinalChange={({ value }) => Streamlit.setComponentValue(value)}
        min={65}
        max={200}
      />
    </>
  );
};

// Link the component to Streamlit JS events.
export default withStreamlitConnection(CustomSlider);
