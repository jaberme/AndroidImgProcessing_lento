package es.ual.bermejo.DemoGoingFaster;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.view.View;
import android.widget.EditText;

public class MatrixActivity extends AppCompatActivity {

    byte[] matrixValues;
    EditText[] editTexts;
    EditText divisor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_matrix);
        editTexts = new EditText[9];
        editTexts[0] = (EditText) this.findViewById(R.id.editText);
        editTexts[1] = (EditText) this.findViewById(R.id.editText2);
        editTexts[2] = (EditText) this.findViewById(R.id.editText3);
        editTexts[3] = (EditText) this.findViewById(R.id.editText4);
        editTexts[4] = (EditText) this.findViewById(R.id.editText5);
        editTexts[5] = (EditText) this.findViewById(R.id.editText6);
        editTexts[6] = (EditText) this.findViewById(R.id.editText7);
        editTexts[7] = (EditText) this.findViewById(R.id.editText8);
        editTexts[8] = (EditText) this.findViewById(R.id.editText9);
        divisor = (EditText) this.findViewById(R.id.divisor);

    }

    public void sendMatrix(View v){
        String temp;
        matrixValues = new byte[9];
        // Guarda los valores de los editTexts en un array
        for(int i=0; i < editTexts.length;i++){
            temp = editTexts[i].getText().toString();
            matrixValues[i]= Byte.parseByte(temp.isEmpty() ? "1" : temp);
        }
        // Envia los resultados a la actividad principal
        Intent intent = getIntent();
        intent.putExtra("matrix", matrixValues);
        String d = divisor.getText().toString();
        intent.putExtra("divisor",Integer.parseInt(!d.isEmpty()?d:"1"));
        setResult(Activity.RESULT_OK, intent);
        finish();
    }

}
