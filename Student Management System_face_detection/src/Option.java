package front;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class Option implements ActionListener
{
	JButton b1;
	JButton b2;
	JButton b3;
	JButton b4;
	JButton b5;
	JButton b6;
	JButton b7;
	JButton b8;

	JFrame frame2;
	Option()
	{
		frame2 = new JFrame("Feature");
		frame2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame2.getContentPane().setBackground( Color.RED);
		JPanel p1= new JPanel(new GridLayout(1,1));
		JPanel p3= new JPanel(new FlowLayout());
		JPanel p2= new JPanel(new GridLayout(3,2));
		b1= new JButton("Attendance");
		b2= new JButton("Student Details");
		b3= new JButton("Result");
		b4= new JButton("Fees");
		b5= new JButton("Libery");
		b6= new JButton("Fine");
		b7= new JButton("Home");
		b8= new JButton("Back");

		JLabel l1=new JLabel("<html><span style='font-size:15px;'>"+"Choose your option"+"</span></html>");
		b7.setHorizontalAlignment(SwingConstants.LEFT);
		b8.setHorizontalAlignment(SwingConstants.CENTER);
		p1.add(l1);
		p2.add(b1);
		p2.add(b2);
		p2.add(b3);
		p2.add(b4);
		p2.add(b5);
		p2.add(b6);
		p3.add(b7);
		p3.add(b8);

		p3.setPreferredSize(new Dimension(300, 300));
		l1.setHorizontalAlignment(JLabel.CENTER);
		b1.addActionListener (this);
		b2.addActionListener (this);
		b3.addActionListener (this);
		b4.addActionListener (this);
		b5.addActionListener (this);
		b6.addActionListener (this);
		b7.addActionListener (this);
		b8.addActionListener (this);
		frame2.add(p1);
		frame2.add(p2);
		frame2.add(p3);
		frame2.setLayout(new GridLayout(3,1));
		frame2.setResizable(false);
		frame2.setSize(500,400);
		frame2.setLocation(200, 100);
		frame2.setVisible(true);
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		// TODO Auto-generated method stub
		if(e.getSource()==b1)
		{
			frame2.setVisible(false);
			new Attendance();
		}
		else if(e.getSource()==b2)
		{
			frame2.setVisible(false);
			new Details();

		}
		else if(e.getSource()==b3)
		{
			frame2.setVisible(false);
			new Result();

		}
		else if(e.getSource()==b4)
		{
			frame2.setVisible(false);
			new Fees();

		}
		else if(e.getSource()==b5)
		{
			frame2.setVisible(false);
			new Library();

		}
		else if(e.getSource()==b6)
		{
			frame2.setVisible(false);
			new Fine();

		}
		else if(e.getSource()==b7)
		{
			frame2.setVisible(false);
			new Home();

		}
		else if(e.getSource()==b8)
		{
			frame2.setVisible(false);
			new Start();

		}

	}
}